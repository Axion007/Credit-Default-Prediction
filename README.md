## 📊 Credit Default Prediction System

![Streamlit App Screenshot](images/capture.png)

## Overview
This project predicts whether a loan applicant is likely to default using machine learning techniques.

## Dataset
The dataset is sourced from the UCI Machine Learning Repository.  
It was collected, cleaned, and preprocessed by the participant for this project.

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest (Final Model)

Random Forest was selected based on higher Recall and F1-score, which are critical for credit risk prediction.

## Deployment
Model training and preprocessing were performed using Google Colab.  
The trained model was exported and deployed locally using Streamlit.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
