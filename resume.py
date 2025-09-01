import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import docx
from PyPDF2 import PdfReader

# Load pre-saved model & data
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("resume_model")   # load saved model
    jobs = joblib.load("jobs_dataset.pkl")
    job_embeddings = joblib.load("job_embeddings.pkl")
    return model, jobs, job_embeddings

model, jobs, job_embeddings = load_model_and_data()

# Function to read resumes
def read_resume(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return None

st.title("📄 AI Resume Screening System (with Saved Model)")
st.write("Upload your resume to see how well it matches job descriptions.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])

if uploaded_file:
    resume_text = read_resume(uploaded_file)
    if resume_text:
        # Encode resume
        resume_emb = model.encode([resume_text])
        # Compute similarity with precomputed job embeddings
        sims = cosine_similarity(resume_emb, job_embeddings)[0]
        jobs['match_score'] = (sims * 100).round(2)

        # Top matches
        top_matches = jobs.sort_values(by="match_score", ascending=False).head(5)

        st.subheader("📊 Top Matching Jobs")
        for _, row in top_matches.iterrows():
            st.markdown(f"""
            **{row['Job Title']}**  
            Match Score: 🎯 {row['match_score']}%  
            **Skills Match:** {row['skills']}  
            """)
    else:
        st.error("Unsupported file format.")
