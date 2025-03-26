import streamlit as st
import os
import requests
import subprocess
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
ASSEMBLY_AI_KEY = os.getenv("ASSEMBLY_AI_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Convert audio to MP3
def convert_audio_to_mp3(input_path, output_path):
    try:
        command = f"ffmpeg -i \"{input_path}\" \"{output_path}\" -y"
        subprocess.run(command, shell=True, check=True)
        return output_path
    except subprocess.CalledProcessError:
        st.error("Audio conversion failed.")
        return None

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
def generate_soap_notes(transcribed_text, patient_name, visit_date):
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
"""

    prompt = f"""
Convert the following doctor-patient conversation into a structured SOAP Note.
Include patient name and visit date in the output.

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
            {"role": "system", "content": "You are a helpful AI medical assistant that generates SOAP notes from doctor-patient conversations."},
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

with st.form("patient_info"):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name")
    with col2:
        visit_date = st.date_input("Date of Visit", value=datetime.today())
    audio_file = st.file_uploader("📤 Upload recorded audio (.m4a or .mp3)", type=["m4a", "mp3"])
    submitted = st.form_submit_button("🚀 Generate SOAP Note")

import tempfile

if submitted and audio_file:
    # Save uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
        tmp_file.write(audio_file.read())
        temp_input_path = tmp_file.name

    mp3_path = temp_input_path.replace(".m4a", ".mp3")
    audio_path = convert_audio_to_mp3(temp_input_path, mp3_path)

    if audio_path:
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_path)

        st.subheader("📝 Transcription:")
        st.write(transcript)

        with st.spinner("Generating SOAP note..."):
            soap_note = generate_soap_notes(transcript, patient_name, visit_date.strftime("%Y-%m-%d"))

        st.subheader("📋 SOAP Note:")
        st.markdown(soap_note.replace("-", "\n-"))
    else:
        st.error("❌ Audio conversion failed.")
