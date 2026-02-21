import streamlit as st
import PyPDF2
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Resume Analyzer Pro", layout="wide")

st.title(" AI Resume Analyzer - Placement Edition")
st.write("Smart ATS-style Resume Evaluation using NLP")

# UI Polish
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description Here")

# Technical skills list
skill_list = [
    "python", "java", "c++", "sql", "machine learning",
    "data structures", "algorithms", "flask", "spring boot",
    "html", "css", "javascript", "react", "azure", "aws",
    "git", "docker", "kubernetes"
]

# ------------- FUNCTIONS ----------------

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
    return " ".join(tokens)

# ------------- MAIN LOGIC ----------------

if uploaded_file is not None and job_description:

    resume_raw = extract_text_from_pdf(uploaded_file)

    resume_text = preprocess(resume_raw)
    job_text = preprocess(job_description)

    # -------- TF-IDF Similarity --------
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors)[0][1]
    match_percentage = round(similarity * 100, 2)

    st.subheader("📊 ATS Match Score")
    st.progress(int(match_percentage))
    st.write(f"### {match_percentage}%")

    # -------- Score Breakdown --------
    skill_score = 0
    project_score = 0
    education_score = 0

    detected_skills = [skill for skill in skill_list if skill in resume_text]
    missing_skills = [skill for skill in skill_list if skill in job_text and skill not in resume_text]

    if detected_skills:
        skill_score = min(len(detected_skills) * 5, 30)

    if "project" in resume_text:
        project_score = 20

    if "btech" in resume_text or "degree" in resume_text:
        education_score = 20

    final_breakdown = skill_score + project_score + education_score

    st.subheader("📌 Score Breakdown")
    st.write(f"Skill Score: {skill_score}/30")
    st.write(f"Project Section Score: {project_score}/20")
    st.write(f"Education Score: {education_score}/20")
    st.write(f"Total Section Score: {final_breakdown}/70")

    # -------- Skill Analysis --------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Skills Found")
        st.write(detected_skills)

    with col2:
        st.subheader("❌ Missing Skills")
        st.write(missing_skills)

    # -------- Skill Frequency Chart --------
    st.subheader("📈 Skill Frequency in Resume")

    skill_counts = {}
    for skill in skill_list:
        skill_counts[skill] = resume_text.count(skill)

    df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
    df = df[df["Count"] > 0]

    if not df.empty:
        fig, ax = plt.subplots()
        ax.bar(df["Skill"], df["Count"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------- Resume Improvement Suggestions --------
    st.subheader("📌 Resume Improvement Suggestions")

    if match_percentage > 80:
        st.success("🔥 Excellent Match! Your resume aligns very well.")
    elif match_percentage > 60:
        st.warning("⚡ Good Match. You can improve further.")
    else:
        st.error("🚨 Low Match. Significant improvement needed.")

    if missing_skills:
        st.write("### 🔧 Add These Important Skills:")
        for skill in missing_skills:
            st.write(f"- {skill}")

    if len(detected_skills) < 5:
        st.write("### 📈 Suggestion:")
        st.write("Try adding more technical projects and measurable achievements.")

    if "project" not in resume_text:
        st.write("###  Add a Projects Section")
        st.write("Include 2–3 strong technical projects with technologies used.")

    # -------- Named Entity Recognition --------
    st.subheader("🏢 Detected Entities (Organizations, Dates, etc.)")
    doc = nlp(resume_raw)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(entities[:10])

    # -------- Download Report --------
    report = f"""
AI Resume Analyzer - Placement Report

ATS Match Score: {match_percentage}%

Skills Found:
{detected_skills}

Missing Skills:
{missing_skills}

Skill Score: {skill_score}/30
Project Score: {project_score}/20
Education Score: {education_score}/20
"""

    st.download_button(
        label=" Download Full Report",
        data=report,
        file_name="placement_resume_report.txt",
        mime="text/plain"
    )