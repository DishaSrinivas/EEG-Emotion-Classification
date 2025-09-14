# EEG Emotion Classification

This project implements an EEG-based emotion classification pipeline using machine learning.  
It performs feature selection, dimensionality reduction, classification, and brain-region level interpretation.

Requirements-
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

 Steps-
- Loads EEG emotion dataset (`emotions.csv`)
- Preprocesses and encodes emotion labels
- Selects top 63 EEG features using Mutual Information
- Performs PCA visualization of EEG feature space
- Trains a Random Forest classifier
- Evaluates performance with:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Plots feature importances
- Maps feature importance to brain regions


