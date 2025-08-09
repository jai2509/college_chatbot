import os
import streamlit as st
from langdetect import detect
import requests

# Load API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Load embedded text files
@st.cache_data
def load_text_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

college_text_ar = load_text_file("embedded_college.txt")
student_guide_text_ar = load_text_file("embedded_student.txt")

knowledge_base = college_text_ar + "\n" + student_guide_text_ar

# Function to generate Groq chat completion
def generate_response(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }
    try:
        resp = requests.post("https://api.groq.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        return f"Error from Groq API: {e}"

# Streamlit UI
st.title("📚 College Enquiry & FAQ Chatbot")
user_input = st.text_input("Ask your question in English or Arabic:")

if user_input:
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    prompt = (
        f"You are a helpful assistant for college enquiries. "
        f"Answer the question using the context below. Respond in {'English' if lang == 'en' else 'Arabic'} only. "
        f"Do not include sources or scores.\n\n"
        f"Context:\n{knowledge_base}\n\nQuestion: {user_input}\nAnswer:"
    )

    with st.spinner("Generating answer..."):
        answer = generate_response(prompt)

    st.markdown(f"### Answer:\n{answer}")
