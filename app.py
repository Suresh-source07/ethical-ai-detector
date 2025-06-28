import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}




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
    
    
    st.subheader("🤖 Choose Model for Training")
    model_name = st.selectbox("Select ML Model", [
        "Logistic Regression", 
        "Decision Tree", 
        "Random Forest", 
        "Support Vector Machine", 
        "K-Nearest Neighbors"
        ])

    if(model_name=='Logistic Regression'):
            model = LogisticRegression(max_iter=1000)
    elif(model_name=='Decision Tree'):
            model = DecisionTreeClassifier()
    elif(model_name=='Random Forest'):
            model = RandomForestClassifier()
    elif(model_name=='Support Vector Machine'):
            model = SVC()
    elif(model_name=='K-Nearest Neighbors'):
            model = KNeighborsClassifier()

    y = df[target_column]
    sensitive = df[sensitive_column]

    # Drop target and sensitive columns from features
    X = df.drop([target_column, sensitive_column], axis=1)

    # Convert categorical features to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Encode target if it's not numeric (important for model training)
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y = y.astype('category').cat.codes

    # Encode sensitive column if it's categorical
    if sensitive.dtype == 'object' or str(sensitive.dtype).startswith('category'):
        sensitive = sensitive.astype('category').cat.codes

    # Handle any NaN or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    sensitive = sensitive.loc[X.index]


    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X, y, sensitive, test_size=0.2, random_state=42
        )

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    sensitive_test = sensitive_test.reset_index(drop=True)


    if "trained" not in st.session_state:
        st.session_state.trained = False
    
    if st.button("Train and Analyze Bias"):
        model.fit(X_train, y_train)
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        y_pred = model.predict(X_test)

        # Store in session_state
        st.session_state.trained = True
        st.session_state.model = model
        st.session_state.shap_values = shap_values
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.sensitive_test = sensitive_test
        st.session_state.sensitive_column = sensitive_column
        st.session_state.df = df
        st.session_state.target_column = target_column

        overall_accuracy = accuracy_score(y_test, y_pred)


        X_test_for_bias = X_test.copy()
        X_test_for_bias[sensitive_column] = sensitive_test

        group_accuracies = []
        for group in X_test_for_bias[sensitive_column].unique():
            idx = X_test_for_bias[X_test_for_bias[sensitive_column] == group].index
            acc = accuracy_score(y_test.loc[idx], y_pred[idx])
            group_accuracies.append((group, acc))

        accuracies_only = [acc for _, acc in group_accuracies]
        bias_gap = max(accuracies_only) - min(accuracies_only)
        variance = np.var(accuracies_only)

        st.subheader("📉 Bias & Variance Summary")

        st.write(f"🎯 Overall Accuracy: `{accuracy_score(y_test, y_pred):.2f}`")
        st.write(f"🔺 **Bias Gap (Max - Min Accuracy):** `{bias_gap:.2f}`")
        st.write(f"📈 **Variance Across Groups:** `{variance:.4f}`")

        if bias_gap > 0.2:
            st.error("⚠️ High bias detected! Consider mitigation strategies.")
        elif bias_gap > 0.1:
            st.warning("⚠️ Moderate bias detected.")
        else:
            st.success("✅ Low bias. Model is performing fairly across groups.")
            
            
        if variance > 0.02:
            st.error("⚠️ High variance across groups – model performance is inconsistent.")
        elif variance > 0.01:
            st.warning("⚠️ Moderate variance across groups.")
        else:
            st.success("✅ Low variance – model is consistent across groups.")
            
        if bias_gap > 0.2 or variance > 0.02:
            st.subheader("🛠️ Bias Mitigation Suggestions")
            st.markdown("""
        - **Re-sampling**: Try oversampling underrepresented groups (like using SMOTE) or undersampling the dominant group.
        - **Reweighting**: Assign different weights to samples from different sensitive groups during model training.
        - **Fair Classifiers**: Use algorithms like `fairlearn`, `aif360`, or `fairboost` that build fairness directly into models.
        - **Feature Removal**: Remove sensitive attributes or correlated proxies if they leak bias.
        - **Threshold Adjustment**: Post-process predictions and adjust thresholds per group to equalize outcomes.
            """)

            st.info("📌 Tip: You can explore packages like `Fairlearn`, `AIF360`, or `FairXGBoost` to implement these methods.")

        
        top_sensitive_values = df[sensitive_column].value_counts().nlargest(10).index
        filtered_df = df[df[sensitive_column].isin(top_sensitive_values)]

        # Convert target to numeric if needed
        if filtered_df[target_column].dtype == 'object':
            filtered_df[target_column] = filtered_df[target_column].astype('category').cat.codes

        # Group and plot
        filtered_df[sensitive_column] = filtered_df[sensitive_column].apply(lambda x: str(x)[:40] + "..." if len(str(x)) > 40 else str(x))

        group_means = filtered_df.groupby(sensitive_column)[target_column].mean()
        st.subheader("📊 Visualization: Sensitive vs Target Column")
        fig, ax = plt.subplots(figsize=(12, 5))
        group_means.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_ylabel(f"Mean {target_column}")
        ax.set_title(f"Mean {target_column} by {sensitive_column}")
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        
        for i, v in enumerate(group_means):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
    if st.session_state.trained:
        shap_values = st.session_state.shap_values
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        sensitive_test = st.session_state.sensitive_test
        sensitive_column = st.session_state.sensitive_column
        df = st.session_state.df
        target_column = st.session_state.target_column

        st.subheader("🔍 Global Feature Importance (SHAP)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        sample_idx = st.slider("Pick a test sample to explain", 0, len(X_test)-1, 0)
        st.subheader(f"🔎 SHAP Explanation for Sample {sample_idx}")
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        st.pyplot(fig2)




    if (st.button("Train and Compare all models")):
        comparison_results=[]
        
        for model_name,model in models.items():
            try:
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test,y_pred)
                
                group_accuracies=[]
                for group in sensitive_test.unique():
                    idx = sensitive_test[sensitive_test==group].index
                    group_acc = accuracy_score(y_test.iloc[idx],y_pred[idx])
                    group_accuracies.append(group_acc)
                    
                bias_gap = max(group_accuracies)-min(group_accuracies)
                variance = np.var(group_accuracies)
                
                comparison_results.append({
                    "Model":model_name , 
                    "Accuracy":round(acc,3),
                    "Bias Gap":round(bias_gap,3),
                    "Variance":round(variance,3),
                })
                
            except Exception as e:
                st.error(f"Model {model_name} failed: {str(e)}")
                
            
        st.subheader("📊 Model Comparison Results")
        result_df = pd.DataFrame(comparison_results)

        if not result_df.empty and "Bias Gap" in result_df.columns:
            st.dataframe(result_df.sort_values(by="Bias Gap", ascending=True).reset_index(drop=True))        

            best_model = result_df.loc[result_df['Bias Gap'].idxmin()]
            st.success(f"Lowest Bias: '{best_model['Model']}' with Bias Gap = '{best_model['Bias Gap']}'")

            st.subheader("📊 Visual Comparison of Models")
            fig, ax = plt.subplots(figsize=(10, 5))
            result_df.plot(x="Model", y=["Bias Gap", "Variance"], kind="bar", ax=ax)
            plt.xticks(rotation=45)
            plt.ylabel("Metric Value")
            plt.title("Bias Gap and Variance Comparison Across Models")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No model comparison data available. All models may have failed.")
            
                
            if not result_df.empty:
                best_model = result_df.loc[result_df['Bias Gap'].idxmin()]
                st.success(f"Lowest Bias:'{best_model['Model']}' with Bias Gap = '{best_model['Bias Gap']}'")
                    
                st.subheader("📊 Visual Comparison of Models")
                fig, ax = plt.subplots(figsize=(10, 5))
                result_df.plot(x="Model", y=["Bias Gap", "Variance"], kind="bar", ax=ax)
                plt.xticks(rotation=45)
                plt.ylabel("Metric Value")
                plt.title("Bias Gap and Variance Comparison Across Models")
                st.pyplot(fig)
