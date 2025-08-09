import os
import streamlit as st
from langdetect import detect
from sentence_transformers import SentenceTransformer
import faiss

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

# Prepare embeddings
@st.cache_resource
def prepare_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = knowledge_base.split("\n")
    vectors = model.encode(docs)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return model, index, docs

model, index, docs = prepare_index()

# Retrieve answer from context
def retrieve_answer(query, lang):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, 3)
    context = "\n".join([docs[i] for i in I[0] if i < len(docs)])
    # Simple heuristic: just return context as answer
    return context

# Streamlit UI
st.title("📚 College Enquiry & FAQ Chatbot (Offline Mode)")
user_input = st.text_input("Ask your question in English or Arabic:")

if user_input:
    try:
        lang = detect(user_input)
    except:
        lang = "en"
    
    with st.spinner("Searching knowledge base..."):
        answer = retrieve_answer(user_input, lang)

    st.markdown(f"### Answer:\n{answer}")
