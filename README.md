# 💳 Credit Card Fraud Detection Web App

This is a Flask-based web application that predicts whether a credit card transaction is **fraudulent or legitimate** using a trained machine learning model. It also displays the model’s accuracy on the provided dataset (`creditcard.csv`).

---

## 🚀 Features

- 🔍 Predicts fraudulent transactions
- 📊 Shows accuracy and ROC-AUC score
- 🧠 Uses a trained RandomForestClassifier with SMOTE for imbalance handling
- 🌐 Web UI with Flask
- 📁 Inputs: 28 anonymized PCA features + transaction Amount

---

## 🧰 Tech Stack

- Python 3.x
- Flask
- NumPy, Pandas
- scikit-learn
- imbalanced-learn (`SMOTE`)
- joblib (for model serialization)

---

## 🧠 Model Training Pipeline

The model is trained using a Random Forest Classifier. Key steps:

1. Load dataset (`creditcard.csv`)
2. Drop unnecessary `Time` column
3. Normalize the `Amount` column using `StandardScaler`
4. Balance the classes using `SMOTE`
5. Split into train and test sets
6. Train using `RandomForestClassifier`
7. Evaluate with `accuracy_score` and `roc_auc_score`
8. Save model as `model.pkl` using `joblib`

## 🗂 Folder Structure

creditcard-fraud-detector/
├── model.pkl # Trained machine learning model
├── creditcard.csv # Sample test dataset (Kaggle dataset)
├── app.py # Flask application code
├── fraud_model_training.ipynb
├── templates/
│ └── index.html # HTML template
├── static/ # (Optional) CSS or JS files
└── README.md # Project documentation

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NikshepPaliwal/CREDIT-CARD-FRAUD-DETECTION.git
cd CREDIT-CARD-FRAUD-DETECTION

Install Required Libraries
pip install -r requirements.txt

If requirements.txt does not exist, install manually:
    pip install flask pandas numpy scikit-learn joblib imbalanced-learn


Run the App
    python app.py

Then open your browser and go to:
👉 http://127.0.0.1:5000/
