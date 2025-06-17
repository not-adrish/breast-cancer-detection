# 🩺 Breast Cancer Diagnosis with LightGBM, PCA & SHAP

This project presents a robust, **high-performing binary classification model** built using the Breast Cancer Wisconsin Diagnostic Dataset. Leveraging **LightGBM**, **automated hyperparameter tuning (Optuna)**, and **explainability via SHAP**, this notebook demonstrates how to construct a production-grade machine learning pipeline that is accurate, interpretable, and reproducible.

---

## 🧠 Why This Matters

Breast cancer is one of the most prevalent cancers globally, affecting **1 in 8 women** during their lifetime. Timely diagnosis can drastically improve survival rates. However, medical professionals continue to face challenges such as:

- **High workload and burnout**
- **Subtle abnormalities missed in early stages**
- **Variability in imaging interpretations**
This can seriously hinder early diagnosis

By combining statistical rigor and machine learning, we can **augment diagnostic workflows** and build tools that provide a "second set of eyes" to support medical practitioners.

This model uses **tabular features derived from digitized imaging** (e.g., texture, radius, compactness), which are often part of early screenings like mammograms or fine needle aspirations.

---

## 📚 Dataset Overview

- Source: `sklearn.datasets.load_breast_cancer`
- Size: **569 samples**, **30 features**
- Task: **Binary Classification** (`malignant` = 0, `benign` = 1)
- Features: Numeric measurements such as `mean radius`, `texture`, `smoothness`, `concavity`, etc.

---

## 🔧 Model Workflow

### ✅ Step-by-step Flowchart

![breast-cancer-flowchart](https://github.com/user-attachments/assets/37fafd17-4016-4499-aa45-bea6d17dcb2a)


---

## 🔬 Methods Used

### 📌 1. **Data Preprocessing**
- Standard scaling of features for numerical stability.
- Optional PCA for **dimensionality reduction** and **visualization**.
- Split into train/test for performance validation.

### 📌 2. **Model: LightGBM Classifier**
- Gradient boosting decision trees, optimized for **speed** and **accuracy**.
- Handles **class imbalance** and works well on tabular data.

### 📌 3. **Hyperparameter Tuning: Optuna**
- Bayesian optimization to maximize F1 Score.
- Tuned parameters: `num_leaves`, `max_depth`, `learning_rate`, `subsample`, etc.
- Visualized importance of hyperparameters.

### 📌 4. **Interpretability: SHAP**
- TreeExplainer used to derive per-feature impact.
- Visualization of global feature importance.

---


### 🔍 Classification Report
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114


