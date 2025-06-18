import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



st.title('Ethical AI Detector')
st.write('Welcome to the Ethical AI Bias Detector – upload a dataset and analyze it for bias and variance.')

uploaded_file = st.file_uploader("Upload your CSV dataset",type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.write(df.head())
    
    st.write(f" Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    
    st.subheader(" Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader(" Column Types")
    st.write(df.dtypes)
    
    st.subheader("Select Target Column")
    target_column = st.selectbox("Choose the target (label) column", df.columns)
    
    if df[target_column].nunique() > 10:
        st.warning("⚠️ Your target column seems continuous. Please choose a classification column (e.g. yes/no, 0/1).")


    st.subheader(" Select Sensitive Column")
    sensitive_column = st.selectbox("Choose the sensitive attribute", df.columns)

    st.success(f"Target: {target_column} | Sensitive Feature: {sensitive_column}")


    if st.button("Train and Analyze Bias"):
            # 1. Separate target and sensitive feature
        y = df[target_column]
        sensitive = df[sensitive_column]

        # 2. Drop target and sensitive column to get features
        X = df.drop([target_column, sensitive_column], axis=1)

        # 3. Convert categorical features to numeric
        X = pd.get_dummies(X, drop_first=True)

        # 4. Encode sensitive column if it's categorical (for grouping)
        if sensitive.dtype == 'object':
            sensitive = sensitive.astype('category').cat.codes
        
        st.success("✅ Data preprocessed successfully! Ready for model training.")
        
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X, y, sensitive, test_size=0.2, random_state=42
        )

        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        sensitive_test = sensitive_test.reset_index(drop=True)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"🎯 **Overall Accuracy:** `{overall_accuracy:.2f}`")
        
        X_test[sensitive_column] = sensitive_test


        group_accuracies = []
        for group in X_test[sensitive_column].unique():
            idx = X_test[X_test[sensitive_column] == group].index
            acc = accuracy_score(y_test.loc[idx], y_pred[idx])
            group_accuracies.append((group, acc))

     
        accuracies_only = [acc for _, acc in group_accuracies]
        bias_gap = max(accuracies_only) - min(accuracies_only)

        st.subheader("📉 Bias Summary")
        st.write(f"🔺 **Bias Gap (Max - Min Accuracy):** `{bias_gap:.2f}`")

        if bias_gap > 0.2:
            st.error("⚠️ High bias detected! Consider rebalancing your dataset or using fairness-aware algorithms.")
        elif bias_gap > 0.1:
            st.warning("⚠️ Moderate bias detected. You may want to investigate further.")
        else:
            st.success("✅ Low bias. The model is performing fairly across groups.")
