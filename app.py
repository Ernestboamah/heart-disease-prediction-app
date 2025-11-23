# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score


# ----------------------
# 1. Page config
# ----------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.sidebar.title("Navigation")

# ----------------------
# 2. Load artifacts
# ----------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")
    columns = joblib.load("columns.pkl")
    df = pd.read_csv("synthetic_heart_disease_dataset.csv")
    categorical_cols = ['Gender', 'Smoking', 'Alcohol_Intake', 'Physical_Activity', 'Diet', 'Stress_Level']
    for col in categorical_cols:
        df[col] = df[col].fillna("None").astype(str)
    # Binary fields as integers
    binary_cols = ['Hypertension', 'Diabetes', 'Hyperlipidemia', 'Family_History', 'Previous_Heart_Attack']
    for col in binary_cols:
        df[col] = df[col].astype(int)
    return model, scaler, threshold, columns, df

lr_model, scaler, thresh, X_columns, df = load_artifacts()

# Identify features
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove('Heart_Disease')
categorical_cols = ['Gender', 'Smoking', 'Alcohol_Intake', 'Physical_Activity', 'Diet', 'Stress_Level']
binary_cols = ['Hypertension', 'Diabetes', 'Hyperlipidemia', 'Family_History', 'Previous_Heart_Attack']

# ----------------------
# 3. Sidebar navigation
# ----------------------
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Data Visualization", "Model Metrics"])

# ----------------------
# 4. Home Page
# ----------------------
if page == "Home":
    st.markdown("## 💓 Heart Disease Dashboard")
    st.markdown("Welcome! Use the sidebar to navigate between pages.")
    
    # Key insights
    st.subheader("Dataset Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", df.shape[0])
    col2.metric("Heart Disease Prevalence", f"{df['Heart_Disease'].mean()*100:.1f}%")
    col3.metric("Average Age", f"{df['Age'].mean():.1f} yrs")
    
    # Pie chart
    heart_counts = df['Heart_Disease'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(heart_counts, labels=["No", "Yes"], autopct='%1.1f%%', colors=["#5DADE2", "#E74C3C"])
    ax.set_title("Heart Disease Distribution")
    st.pyplot(fig)

# ----------------------
# 5. Prediction Page
# ----------------------
elif page == "Prediction":
    st.markdown("## 🩺 Predict Heart Disease")
    st.markdown("Adjust the values below to predict risk for a new patient.")

    # Layout with 3 columns
    col1, col2, col3 = st.columns(3)
    user_input = {}

    # Numeric inputs
    for i, col in enumerate(numeric_cols):
        column = [col1, col2, col3][i % 3]
        if pd.api.types.is_integer_dtype(df[col]):
            user_input[col] = column.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].mean()))
        else:
            user_input[col] = column.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()), step=0.1)

    # Binary inputs
    for i, col in enumerate(binary_cols):
        column = [col1, col2, col3][i % 3]
        user_input[col] = 1 if column.selectbox(col, ["No", "Yes"])=="Yes" else 0

    # Categorical inputs
    for i, col in enumerate(categorical_cols):
        column = [col1, col2, col3][i % 3]
        user_input[col] = column.selectbox(col, df[col].unique())

    # Convert to dataframe and encode
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    missing_cols = set(X_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X_columns]

    # Scale
    input_scaled = scaler.transform(input_encoded)
    pred_prob = lr_model.predict_proba(input_scaled)[:,1][0]
    pred_class = int(pred_prob > thresh)

    # Display
    st.subheader("Prediction Result")
    if pred_class == 1:
        st.error(f"💓 High Risk: Heart Disease predicted ({pred_prob:.2f})")
    else:
        st.success(f"💓 Low Risk: Heart Disease predicted ({pred_prob:.2f})")
    st.progress(pred_prob)
    st.info(f"Threshold used: {thresh:.2f}")

# ----------------------
# 6. Data Visualization
# ----------------------
elif page == "Data Visualization":
    st.markdown("## 📊 Data Visualization")
    feature_type = st.radio("Select Feature Type:", ["Numeric", "Categorical"])
    
    if feature_type == "Numeric":
        selected_feature = st.selectbox("Select Numeric Feature", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, color="#5DADE2", ax=ax)
        st.pyplot(fig)
    else:
        selected_feature = st.selectbox("Select Categorical Feature", categorical_cols + binary_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=selected_feature, data=df, palette=["#5DADE2", "#E74C3C"], ax=ax)
        st.pyplot(fig)

# ----------------------
# 7. Model Metrics
# ----------------------
elif page == "Model Metrics":
    st.markdown("## 📈 Model Metrics")
    X_full = pd.get_dummies(df.drop("Heart_Disease", axis=1))
    missing_cols = set(X_columns) - set(X_full.columns)
    for col in missing_cols:
        X_full[col] = 0
    X_full = X_full[X_columns]
    y = df['Heart_Disease']
    X_scaled = scaler.transform(X_full)
    y_pred_prob = lr_model.predict_proba(X_scaled)[:,1]
    y_pred = (y_pred_prob > thresh)

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{(y_pred==y).mean():.2f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y, y_pred):.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    auc_score = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'LR Model (AUC={auc_score:.2f})', color="#007C91")
    ax.plot([0,1],[0,1],'k--', label='Random Guess')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

