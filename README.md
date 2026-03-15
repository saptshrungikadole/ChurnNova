# 🚀 ChurnNova — AI Customer Churn Intelligence & Retention Platform

ChurnNova is a machine learning-powered web application built with **Streamlit** that predicts customer churn, segments risk levels, and provides actionable retention insights.

---

## 📁 Project Structure

```
ChurnNova/
├── Data/
│   └── Churn.csv                  # Raw customer dataset
├── Model/
│   ├── churn_model.pkl            # Trained Random Forest model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   └── feature_columns.pkl        # Saved feature column names
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/churnnova.git
cd churnnova
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Before launching the app, train the model to generate the required `.pkl` artifacts.

```bash
python train_model.py
```

This will:
- Load and clean `Data/Churn.csv`
- Encode categorical features using one-hot encoding
- Scale features using `StandardScaler`
- Train a `RandomForestClassifier` (300 trees, max depth 12)
- Save `churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` inside the `Model/` folder

**Expected output:**
```
Model Accuracy: XX.XX %
Model, scaler, and features saved successfully.
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ App Features

### 🔮 Tab 1 — Predict Customer
- Input individual customer details (gender, tenure, contract type, charges, etc.)
- Get real-time churn probability with a gauge chart
- Color-coded result: ✅ Likely to Stay or ⚠ High Churn Risk

### 📂 Tab 2 — Bulk Prediction
- Upload a CSV file with multiple customers
- Get churn predictions and probabilities for all rows
- Risk segmentation: **Low / Medium / High Risk**
- Visualize churn risk distribution and pie chart breakdown

### 📊 Tab 3 — Churn Analytics
- Bar charts showing churn rates by:
  - Contract type (Month-to-month, One year, Two year)
  - Internet service type (DSL, Fiber, No Internet)

### 🤖 Tab 4 — Model Intelligence
- Top 15 most influential features (bar chart)
- AI insight highlighting the #1 churn driver
- Model summary (type, feature count)

---

## 📊 Dataset

The model is trained on the **Telco Customer Churn** dataset. Place it at `Data/Churn.csv`.

Expected columns include:

| Column | Description |
|---|---|
| `customerID` | Unique customer identifier (dropped during training) |
| `gender` | Male / Female |
| `SeniorCitizen` | Yes / No |
| `Partner` | Yes / No |
| `Dependents` | Yes / No |
| `tenure` | Months with the company |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Month-to-month / One year / Two year |
| `Churn` | Target variable — Yes / No |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Streamlit | Web UI framework |
| scikit-learn | ML model & preprocessing |
| pandas | Data manipulation |
| Plotly | Interactive charts |
| pickle | Model serialization |

---

## 👤 Author

**Saptshrungi Kadole** — Customer Retention Intelligence System
