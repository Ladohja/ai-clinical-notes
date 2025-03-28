import streamlit as st
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import tempfile
import base64

# Load environment variables
load_dotenv()
ASSEMBLY_AI_KEY = os.getenv("ASSEMBLY_AI_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Transcribe audio with AssemblyAI
def transcribe_audio(audio_path):
    headers = {"authorization": ASSEMBLY_AI_KEY}
    upload_url = "https://api.assemblyai.com/v2/upload"
    with open(audio_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, files={"file": f})
    audio_url = response.json()["upload_url"]

    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    transcript_response = requests.post(transcript_endpoint, json={"audio_url": audio_url, "speaker_labels": True}, headers=headers)
    transcript_id = transcript_response.json()["id"]

    while True:
        polling = requests.get(f"{transcript_endpoint}/{transcript_id}", headers=headers).json()
        if polling['status'] == 'completed':
            return polling['text']
        elif polling['status'] == 'failed':
            return "Transcription failed."

# Generate SOAP Note using Perplexity + Few-shot Example
def generate_soap_notes(transcribed_text, patient_name, visit_date, doctor_name, doctor_specialty):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    few_shot_example = """
Example Conversation:
Patient: I've been feeling very tired lately.
Doctor: How long has this been happening?
Patient: For about a week. I also have a headache.
Doctor: Any fever?
Patient: A slight fever yesterday.

Example SOAP Note:
- Subjective: Patient reports fatigue for one week and headache. Slight fever occurred yesterday.
- Objective: No objective findings mentioned.
- Assessment: Possible viral illness.
- Plan: Recommend rest, fluids, and follow-up if symptoms worsen.
- Follow-up: Yes, if symptoms worsen.
"""

    prompt = f"""
Convert the following doctor-patient conversation into a structured SOAP Note.
Include patient name, visit date, doctor name, and specialty in the output.
If a SOAP section cannot be inferred from the conversation, include "Not Mentioned" for that section. Do not hallucinate information or use external knowledge. Only use what the doctor said.
At the end, include a line for Follow-up recommendation. If unsure, say "Follow-up: Not mentioned."

Doctor Name: {doctor_name}
Specialty: {doctor_specialty}
Patient Name: {patient_name}
Date of Visit: {visit_date}

{few_shot_example}

Conversation:
{transcribed_text}

Return only the SOAP Note.
"""

    body = {
        "model": "sonar-pro",  
        "messages": [
            {"role": "system", "content": "You are a helpful AI medical assistant that generates SOAP notes from doctor-patient conversations. SOAP refers to Subjective, Objective, Assessment, Plan – not messaging protocols."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
    result = response.json()

    st.subheader("🔎 Perplexity API Raw Response:")
    st.json(result)

    try:
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: No choices returned from Perplexity API.\n{e}"

# Streamlit UI Starts Here
st.set_page_config(page_title="AI Clinical Note Generator", layout="centered")
st.title("🩺 AI-Powered Clinical Note Generator")

st.sidebar.markdown("## Customize")
st.sidebar.toggle("Wide Mode")

st.info("Note: All SOAP notes are generated solely from the audio conversation.")
st.warning("Disclaimer: This note reflects only the doctor's perspective and not the patient's.")

with st.form("patient_info"):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
        doctor_name = st.text_input("Doctor Name")
    with col2:
        doctor_specialty = st.text_input("Doctor Specialty")
        visit_date = st.date_input("Date of Visit", value=datetime.today())
    audio_file = st.file_uploader("📤 Upload recorded audio (.mp3 only)", type=["mp3"])
    submitted = st.form_submit_button("🚀 Generate SOAP Note")

if submitted and audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name

    with st.spinner("⏳ Transcribing audio..."):
        transcript = transcribe_audio(audio_path)

    if "failed" not in transcript.lower():
        st.success("✅ Transcription completed!")
        st.subheader("📝 Transcription:")
        st.write(transcript)

        with st.spinner("🔄 Generating SOAP note..."):
            soap_note = generate_soap_notes(transcript, patient_name, visit_date.strftime("%Y-%m-%d"), doctor_name, doctor_specialty)

        st.success("📄 SOAP Note ready!")
        st.subheader("📋 SOAP Note:")
        st.markdown(soap_note.replace("-", "\n-"))

        # Download button
        b64 = base64.b64encode(soap_note.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="soap_note.txt">📥 Download SOAP Note</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.toast("SOAP note ready to download!", icon="🎉")
    else:
        st.error("❌ Transcription failed.")
