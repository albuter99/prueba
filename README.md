# 🌬️ Wind Energy Prediction MLOps Pipeline

## 🚀 Project Overview

This project builds a complete MLOps pipeline to predict theoretical wind energy output based on weather data for Aalborg, Denmark.

The pipeline includes:
* **Data ingestion** from Open-Meteo API.
* **Data preprocessing** and cleaning.
* **Feature engineering** (lags, rolling averages, cubic wind relation).
* **Model training** (Linear Regression & Random Forest).
* **Prediction pipeline**.
* **API deployment** with FastAPI.
* **Frontend visualization** using GitHub Pages.
* **Automation** with GitHub Actions.
* **Monitoring and logging**.
* **Docker containerization**.

---

## 📂 Project Structure

```text
.
├── app/
│   ├── fetch.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── monitoring.py
│   ├── main.py
│   └── api.py
├── artifacts/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   ├── models/
│   ├── predictions/
│   └── metrics/
├── docs/
│   ├── index.html
│   └── predictions.json
├── .github/workflows/
│   └── pipeline.yml
├── Dockerfile
└── requirements.txt

👤 Authors
Ioannis Chatzikos
Kristjana Prifti
Timo Bertus Rik Philipse
Victor Carmona García
Álvaro Buendía 
