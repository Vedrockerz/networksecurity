# 🛡️ Phishing Website Detection System

An end-to-end **Machine Learning project** to classify websites as **phishing or legitimate**, built as part of my learning journey in ML and backend development.  
The focus of this project is on creating a **production-style ML pipeline** and serving predictions using **FastAPI**.

---

## 📌 Problem Statement
Phishing websites pose serious cybersecurity risks by deceiving users into sharing sensitive information.  
This project detects phishing websites using URL-based and domain-based features.

---

## 🏗️ Architecture Overview
Phishing Dataset → MongoDB → Data Ingestion → Data Validation → Data Transformation →  
Model Training → Model Evaluation → Model Artifacts → FastAPI → Prediction Output

---

## ⚙️ Tech Stack
- Python  
- FastAPI  
- MongoDB  
- Scikit-learn  
- MLflow  
- Docker (planned) | AWS / Azure (planned)

---

## 📂 Dataset
- Public phishing website dataset (~11,000 samples)
- 30 URL & domain-based features
- Target column: `Result`
  - `1` → Legitimate  
  - `-1` → Phishing  

Used for educational and learning purposes.

---

## 🔁 ML Pipeline
- Data ingestion from MongoDB  
- Schema & data validation  
- Feature transformation  
- Model training & evaluation  
- Model artifact management  

---

## 🚀 FastAPI Endpoints
Access Swagger UI:  
http://127.0.0.1:8000/docs

- `GET /train` → Train model  
- `POST /predict` → Predict phishing website  

---

## ▶️ Run Locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
uvicorn app:app --reload
```
---
## ⚠️ Disclaimer

This project was built as part of a learning process with course guidance and focuses on understanding real-world ML system design.
