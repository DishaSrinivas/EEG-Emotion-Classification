# EEG Emotion Classification

This project implements an EEG-based emotion classification pipeline using machine learning.  
It performs feature selection, dimensionality reduction, classification, and brain-region level interpretation.

Requirements-
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Data - Dataset can be downloaded from: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data

Citation- Bird, Jordan & Ekart, Aniko & Buckingham, Christopher & Resende Faria, Diego. (2019). Mental Emotional Sentiment Classification with an EEG-based Brain-machine Interface.


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


