# Maritime Hackathon 2025 - Vessel Deficiency Severity Prediction

This project was developed for the **Maritime Hackathon 2025** by Team Duck 🦆.  
It aims to predict vessel deficiency severity using a combination of machine learning, natural language processing (NLP), and structured text parsing.

---

## 📌 Project Overview
The goal of this project is to automate the classification of vessel deficiencies based on inspection reports.  
The model is trained to analyze textual deficiency descriptions and classify them into four severity levels:

- **High**
- **Medium**
- **Low**
- **Not a deficiency**

The implementation integrates NLP techniques, heuristic-based rule systems, and deep learning models to ensure high accuracy in severity prediction.

---

## 🚀 Features & Implementation
### **1. Data Preprocessing**
- Extracted structured fields from deficiency descriptions (e.g., `"Immediate Causes"`, `"Root Cause Analysis"`).
- Applied keyword-based heuristics to assign preliminary severity scores.
- Performed **majority voting** among multiple annotators to derive consensus severity labels.

### **2. Consensus-Based Severity Calculation**
- Used a **majority voting** approach to determine the final severity label from multiple annotations.
- In case of a tie, priority was assigned in the order:  
  `High > Medium > Low > Not a deficiency`.

### **3. Machine Learning Model - DeBERTa Transformer**
- Used **Microsoft DeBERTa-v3** for text-based classification.
- Tokenized input text using **Hugging Face's Transformers** library.
- Trained on labeled deficiency reports using PyTorch.

### **4. Feature Engineering**
- Extracted numerical features such as:
  - Text length
  - Keyword occurrences (e.g., `"critical"`, `"urgent"`, `"minor"`)
  - Domain-specific heuristic signals (e.g., `"immediate cause"`)
- Combined structured textual data with deep learning model predictions.

### **5. Model Training & Optimization**
- Split data into **80% training** and **20% validation**.
- Used **Optuna** for hyperparameter tuning.
- Applied **label smoothing** to improve generalization.
- Trained for **5 epochs** with batch size of **4**.

### **6. Evaluation Metrics**
- Used **Accuracy, Precision, Recall, and F1-score**.
- Evaluated both **text-only model** and **hybrid model (text + features)**.
- Compared against a **baseline heuristic model**.

### **7. Deployment & Usability**
- Model saved and ready for deployment.
- Provides an API-ready function to predict severity from new deficiency reports.

---

## 🔧 Installation & Setup
### **Prerequisites**
Ensure you have **Python 3.8+** installed along with required dependencies.

Clone the repository:
```sh
git clone https://github.com/JellyPenguinnn/Maritime-Hackathon-2025.git
cd Maritime-Hackathon-2025
```

Install dependencies
```sh
pip  install -r requirements.txt
```

## 📈 Model Performance
- Final model achieves high accuracy on validation data
- Performs significantly better than rule-based heuristic models
- Demonstrates robust generalization accross unseen deficiency reports




