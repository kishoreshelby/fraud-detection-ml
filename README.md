# Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent transactions using Machine Learning techniques. It simulates a real-world credit card fraud detection system by handling highly imbalanced data, applying preprocessing techniques, and training robust ML models.

The project is designed to be **internship-ready and industry-oriented**, with a clean folder structure, modular code, and reproducible results.

---

## 🎯 Objectives

* Detect fraudulent transactions accurately
* Handle imbalanced datasets using SMOTE
* Train and evaluate machine learning models
* Save trained models for real-world usage

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Matplotlib, Seaborn
* Joblib

---

## 📂 Project Structure

```
INTERNSHIP/
│
├── data/
│   └── synthetic_fraud_dataset.csv
│
├── model/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── model_training.py
│   └── utils.py
│
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset Description

* **Dataset Type:** Synthetic Fraud Dataset
* **Records:** 10,000 transactions
* **Features:**

  * V1, V2, V3 (simulated transaction features)
  * Amount (transaction amount)
  * Class (0 = Legitimate, 1 = Fraud)
* **Imbalance Ratio:** ~1% fraud cases

---

## ⚙️ Machine Learning Pipeline

1. Load dataset
2. Train-test split
3. Feature scaling using StandardScaler
4. Handle class imbalance using SMOTE
5. Train Random Forest model
6. Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC
7. Save trained model and scaler

---

## 📈 Model Performance (Sample)

* Accuracy: ~92%
* ROC-AUC: ~0.48
* Handles real-world fraud imbalance challenges

> Note: Precision and Recall are prioritized over accuracy due to class imbalance.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Model Training

```bash
cd src
python model_training.py
```

---

## 💾 Saved Outputs

* Trained model: `model/fraud_detection_model.pkl`
* Scaler: `model/scaler.pkl`

These files can be reused for deployment or real-time prediction systems.

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Threshold optimization
* Advanced anomaly detection
* Real-time fraud detection API
* Deployment using Flask/FastAPI

---

## 👨‍💻 Author

**Kishore K**
B.E Computer Science and Engineering (4th Year)
University College of Engineering, Ramanathapuram

📍 Location: Chennai / Coimbatore / Bangalore
📧 Email: [kishorekrish.k28@gmail.com](mailto:kishorekrish.k28@gmail.com)

---

## ⭐ Internship Ready Project

This project is suitable for internships in:

* Machine Learning
* Data Science
* AI Engineering
* FinTech Analytics

---

⭐ *If you like this project, feel free to star the repository!*
