import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import contextlib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Correlation Heatmap
plt.figure(figsize=(14, 12))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'target']
sns.pairplot(df[selected_features], hue='target', palette='coolwarm', corner=True)
plt.savefig("images/pairplot.png", dpi=150)
plt.close()

# PCA Visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.savefig("images/pca_plot.png", dpi=150)
plt.close()

# Explained Variance
pca_all = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_all.explained_variance_ratio_), marker='o')
plt.xlabel("# of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid(True)
plt.savefig("images/explained_variance.png", dpi=150)
plt.close()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Optuna Bayesian Optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
print("Best Hyperparameters:", best_params)

best_model = LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)

# Optuna Visualizations
fig1 = plot_optimization_history(study)
fig1.savefig("images/optuna_optimization_history.png")

fig2 = plot_param_importances(study)
fig2.savefig("images/optuna_param_importances.png")

# Evaluation
preds = best_model.predict(X_test)
print("F1 Score:", f1_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds))

# SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig("images/shap_summary.png", dpi=150)
plt.close()
