## Live Demo

https://ai-resume-analyzer-7mrumhgdogkgh7mywsrgbk.streamlit.app/
#  AI Resume Analyzer – Placement Edition

An NLP-powered ATS (Applicant Tracking System) Resume Analyzer built using Python, spaCy, and Streamlit.

This project compares a candidate's resume with a job description and generates:
- ATS Match Score
- Skill Analysis
- Section-wise Breakdown
- Resume Improvement Suggestions
- Named Entity Recognition
- Downloadable Report

---

## Features

NLP Text Preprocessing using spaCy  
Lemmatization & Stopword Removal  
TF-IDF Vectorization  
Cosine Similarity Matching  
Technical Skill Extraction  
Section-wise Score Breakdown  
Resume Improvement Suggestions  
Named Entity Recognition (NER)  
Skill Frequency Visualization  
Downloadable Analysis Report  

---

##  How It Works

1. Resume is uploaded in PDF format.
2. Text is extracted using PyPDF2.
3. spaCy performs:
   - Tokenization
   - Lemmatization
   - Stopword Removal
4. TF-IDF Vectorization converts text into numerical vectors.
5. Cosine Similarity calculates alignment score between resume and job description.
6. Skills are detected and missing skills are highlighted.
7. A section-wise score is generated.
8. Suggestions are provided to improve resume quality.

---

##  Tech Stack

- Python
- Streamlit
- spaCy
- scikit-learn
- PyPDF2
- Pandas
- Matplotlib

---

##  System Architecture

Resume PDF → Text Extraction → NLP Preprocessing → TF-IDF Vectorization → Cosine Similarity → Skill Detection → Score Generation → Suggestions → Report Download

---

##  ATS Score Breakdown

The system evaluates:

- Technical Skills (30 points)
- Projects Section (20 points)
- Education Section (20 points)

Total Section Score: 70 Points

---

## Installation

Clone the repository:
