# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 10:47:05 2025

@author: Lenovo
"""

"""
EEG Emotion Classification Pipeline
Author: Disha Srinivas
Description:
    This script performs EEG-based emotion classification using:
    - Feature selection (Mutual Information & ANOVA)
    - PCA visualization
    - Random Forest classification
    - Feature importance analysis by brain regions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------
# 1. Data Loading
# -------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load EEG dataset from a CSV file and remove missing values.
    """
    df = pd.read_csv(file_path)
    return df.dropna()


# -------------------
# 2. Preprocessing
# -------------------
def preprocess_data(df: pd.DataFrame):
    """
    Separate features and labels, encode labels, and return encoded data.
    """
    X = df.drop(columns=['label'])
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, y, le


# -------------------
# 3. Feature Selection
# -------------------
def select_top_features(X, y_encoded, k=63):
    """
    Select top k features based on Mutual Information.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y_encoded)
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    return X_selected, selected_features


# -------------------
# 4. PCA Visualization
# -------------------
def plot_pca(X_selected, y, title="PCA: EEG Feature Space"):
    """
    Perform PCA and plot results in 2D.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['label'] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='label', palette='Set2', s=70)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(title='Emotion')
    plt.tight_layout()
    plt.show()


# -------------------
# 5. Model Training
# -------------------
def train_random_forest(X_selected, y_encoded):
    """
    Train a Random Forest classifier and return the trained model and predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return rf, y_pred


# -------------------
# 6. Feature Importance Plot
# -------------------
def plot_feature_importances(rf, selected_features):
    """
    Plot feature importances from Random Forest.
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_features)), importances[indices], align='center')
    plt.xticks(range(len(selected_features)),
               np.array(selected_features)[indices],
               rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.show()


# -------------------
# 7. Brain Region Mapping
# -------------------
def plot_brain_region_importance(rf, selected_features):
    """
    Aggregate feature importances by brain region and plot.
    """
    suffix_to_channel = {
        "_a": "TP9", "_a1": "TP9", "_a2": "TP9",
        "_b": "AF7", "_b1": "AF7", "_b2": "AF7",
        "_c": "AF8", "_c1": "AF8", "_c2": "AF8",
        "_d": "TP10", "_d1": "TP10", "_d2": "TP10"
    }

    brain_region_map = {
        "TP9": "Left Temporal-Parietal",
        "AF7": "Left Frontal",
        "AF8": "Right Frontal",
        "TP10": "Right Temporal-Parietal"
    }

    region_importance = {region: 0 for region in brain_region_map.values()}

    for feat_name, importance in zip(selected_features, rf.feature_importances_):
        for suffix, channel in suffix_to_channel.items():
            if feat_name.endswith(suffix):
                region_importance[brain_region_map[channel]] += importance
                break

    plt.figure(figsize=(8, 6))
    plt.bar(region_importance.keys(), region_importance.values(), color='skyblue')
    plt.ylabel("Total Importance")
    plt.title("Brain Region Contribution to Emotion Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------
# Main Execution
# -------------------
def main():
    file_path = "C:\\Users\\Lenovo\\Desktop\\EEGProject\\data\\emotions.csv"

    # Load & preprocess
    df = load_data(file_path)
    X, y_encoded, y, _ = preprocess_data(df)

    # Feature selection
    X_selected, selected_features = select_top_features(X, y_encoded)

    print("Top features selected:", selected_features)

    # PCA plot
    plot_pca(X_selected, y)

    # Model training
    rf, y_pred = train_random_forest(X_selected, y_encoded)

    # Feature importance
    plot_feature_importances(rf, selected_features)

    # Brain region importance
    plot_brain_region_importance(rf, selected_features)


if __name__ == "__main__":
    main()
