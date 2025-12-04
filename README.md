# 🚦 Real-Time Traffic Forecasting in Paris Using XGBoost & Flask

This project implements a **real-time traffic forecasting system** for two major Paris road arcs (Arc **1727** and Arc **6670**, Porte d’Aubervilliers).
It uses **machine learning (XGBoost)**, **live API data**, and a **Flask web server** to predict traffic for the next 1, 2, and 3 hours.

---

## 📌 Overview

Traffic in urban environments changes rapidly and is influenced by complex temporal patterns.
Traditional models are not sufficient to capture these dynamics.

This project provides:

* ✔️ Machine learning prediction models
* ✔️ Real-time data ingestion from Open Data Paris
* ✔️ REST API built with Flask
* ✔️ A lightweight prediction dashboard

It is **production-ready**, modular, and easy to deploy.

---

## ✨ Features

### 🧠 Machine Learning

* XGBoost models trained for:

  * **1-hour ahead**
  * **2-hours ahead**
  * **3-hours ahead**
* Advanced feature engineering:

  * time-based features (hour, day, month)
  * cyclic encoding (sin/cos)
  * lag features (3 to 6 values)
  * rolling averages (3h, 6h)
  * cross-arc features

### 🌐 Real-Time API Integration

* Fetches data from **Open Data Paris – Traffic Counters**
* Automatically caches data for 1 hour
* Works even if the API becomes unavailable

### 🚀 Flask REST API

Endpoints include:

| Endpoint           | Description                            |
| ------------------ | -------------------------------------- |
| `/api/predict`     | Predict traffic for Now, +1h, +2h, +3h |
| `/api/status`      | Check API status + cache info          |
| `/api/raw-data`    | View raw processed API data            |
| `/api/debug`       | Inspect models + sample data           |
| `/api/clear-cache` | Delete cached data                     |

### 🖥️ Web Dashboard

* Simple interface built with Flask templates
* Buttons for each prediction horizon
* Displays prediction results + source of data (API or cache)

---

## 📊 Tech Stack

* Python 3.x
* Flask
* XGBoost
* Pandas, NumPy
* Requests
* Joblib

---

## 📁 Project Structure

```
/saved_models
    xgb_model_1727.pkl
    xgb_model_6670.pkl
    xgb_model_1727_h2.pkl
    xgb_model_6670_h2.pkl
    xgb_model_1727_h3.pkl
    xgb_model_6670_h3.pkl

/templates
    index.html

app.py
requirements.txt
temp_traffic_cache.csv  (generated automatically)
```

---

## ⚡ How It Works

1. The application requests the last 96 traffic records from the Open Data Paris API.
2. Data is cleaned, sorted, and enhanced through feature engineering.
3. Depending on the chosen horizon, the appropriate XGBoost model is loaded.
4. Prediction is generated and returned in JSON format or displayed on the dashboard.
5. If the Paris API is unavailable, cached data is used as a fallback.

---

## 🧪 Model Performance

* **1h ahead:** High accuracy
* **2h ahead:** Small decrease in precision
* **3h ahead:** Higher error (normal for longer forecasting horizon)

XGBoost provides strong short-term forecasting performance.

---

## ▶️ Run the Project

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the server:

```bash
python app.py
```

### 3. Open the dashboard in your browser:

```
http://127.0.0.1:5000
```

---

## 🔧 Environment Variables (Optional)

If Open Data Paris API changes, you may add config variables such as:

```
API_URL=""
CACHE_TIMEOUT=3600
```

---

## 🚀 Possible Improvements

* Add **LSTM / GRU / Transformer** models
* Multi-step 24-hour forecasting
* Weather-based prediction
* Automatic anomaly detection
* Generalisation for all Paris arcs
