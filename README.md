# 🩺 Breast Cancer Diagnosis with Bayesian Optimized LightGBM 

This project presents a high-performing binary classification model built using the Breast Cancer Wisconsin Diagnostic Dataset. Leveraging **LightGBM**, **automated hyperparameter tuning (Optuna)**, and **SHAP**, this notebook shows how to construct a machine learning pipeline that is accurate, interpretable, and even reproducible.

---

##  Why This Matters

Breast cancer is one of the most prevalent cancers in the whole world, affecting **1 in 8 women** during their lifetime. Timely diagnosis can drastically improve survival rates. However, medical professionals continue to face challenges such as:

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
- Columns: mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension',
       'target'
---

##  Model Workflow

###  Step-by-step Flowchart

![breast-cancer-flowchart](https://github.com/user-attachments/assets/37fafd17-4016-4499-aa45-bea6d17dcb2a)


---

## 🔬 Methods Used

### - 1. **Data Preprocessing**
- Standard scaling of features for numerical stability.
- PCA for **visualization**.
- Split into train/test for performance validation.

### - 2. **Model: LightGBM Classifier**
- Gradient boosting decision trees, optimized for **speed** and **accuracy**.
- Handles **class imbalance** and works well on tabular data.

### - 3. **Hyperparameter Tuning: Optuna**
- Bayesian optimization to maximize F1 Score.
- Tuned parameters: `num_leaves`, `max_depth`, `learning_rate`, `subsample`, etc.
- Visualized importance of hyperparameters.

### - 4. **Interpretability: SHAP**
- TreeExplainer used to derive per-feature impact.
- Visualization of global feature importance.

---

## 🤖 Model Details

### 📊 Model Parameters

| Field                     | Value                      |
|--------------------------|----------------------------|
| Algorithm                | LightGBM Classifier        |
| boosting_type               | 'gbdt'                           |
| num_leaves           | 94          |
| max_depth           | 3      |
| learning_rate       | 0.06119661695834652                        |
| subsample_for_bin         | 200000                        |
| class_weight  | None    |
| n_estimators   |  475                          |
| min_split_gain   | 0.0                           |
| min_child_weight    | 0.001                       |
| mean_child_samples            |    23                       |
| subsample         | 0.5810233282559958                           |
| subsample_freq    | 0                     |
| colsample_bytree | 	0.5765318146443195 |
| reg_alpha |	0.6424818973380652          |
| reg_lambda |	0.0002996507756344334            |
| random_state     | None       |
| n_jobs  | None                   |
| importance_type                 | 'split'                       


### 🔍 Classification Report
```
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

```
---

## Findings

### 2D Projection of Breast Cancer Data via Scatter Plot
![2d_projection_of_breast_cancer_data](https://github.com/user-attachments/assets/6f9c4a92-dbf6-4275-a54f-cf69639c1a98)

### Correlation Heatmap
![corr_heatmap](https://github.com/user-attachments/assets/48419cda-4836-48e3-8929-4d377088b0dc)

# Pairplot of Key Features
![pairplot](https://github.com/user-attachments/assets/c1d48969-d8f9-42f6-a023-6caa7ff2b89c)

### SHAP Feature Importances
![feature_importances](https://github.com/user-attachments/assets/42eeb9be-440b-415e-9e91-f3428a06c6a8)

### Hyperparameter Importances
![hyperparam_importances](https://github.com/user-attachments/assets/6568fac2-fbfd-4824-b3e2-5c45cb2761c9)

### Optuna Optimization History
![optimization_history](https://github.com/user-attachments/assets/ec4b287b-5fe2-41ef-a15f-f439f6b19706)
---

## Disclaimer:
This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice or diagnosis. Always consult qualified healthcare professionals for clinical decisions.

## License:
This project is licensed under the MIT license

## Contributions:
Suggestions, forks, and improvements are welcome! If you have ideas to expand this work, feel free to submit a pull request.

# Author: Adrish Das
---
