# parkinsons_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")

# Load dataset and trained model
data = pd.read_csv('parkinson data.csv')
model = joblib.load('parkinson_detector.pkl')

# Define all features used during training
full_feature_list = data.drop(columns=['status', 'name']).columns.tolist()

# Define top 8 features for user input
top_features = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'NHR', 'Jitter:DDP', 'Shimmer:APQ5', 'MDVP:Shimmer', 'RPDE']

# Streamlit UI
st.title("Parkinson's Disease Prediction App")
st.markdown("This app predicts the **likelihood of Parkinson’s disease** based on key voice features. Trained with all 22 voice biomarkers for maximum accuracy.")

st.sidebar.header("Input Patient Voice Features")

# Collect input for top features
def user_input_features():
    user_input = {}
    for feature in top_features:
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())
        user_input[feature] = st.sidebar.slider(
            feature, min_value=min_val, max_value=max_val, value=mean_val)
    return pd.DataFrame([user_input])

# Get user input
input_df = user_input_features()

# Fill in missing features with column means
for feature in full_feature_list:
    if feature not in input_df.columns:
        input_df[feature] = data[feature].mean()

# Reorder columns to match model
input_df = input_df[full_feature_list]

# Predict
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

# Output
st.subheader("Prediction Result")

if prediction == 1:
    st.error("**Parkinson's Disease Detected**")
else:
    st.success("**Likely Healthy**")

if prediction_proba is not None:
    st.info(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")

# Display input summary
st.subheader("Your Input Summary")
st.write(input_df[top_features])

# --- Visualizations ---
st.markdown("---")
st.markdown("### Data Visualizations")

with st.expander("Correlation Heatmap"):
    fig, ax = plt.subplots()
    sns.heatmap(data.drop(columns=['status', 'name']).corr(), annot=False, cmap="coolwarm")
    st.pyplot(fig)

with st.expander("Distribution: MDVP:Fo(Hz)"):
    fig2, ax2 = plt.subplots()
    sns.histplot(data["MDVP:Fo(Hz)"], kde=True, color="skyblue", bins=30)
    st.pyplot(fig2)

with st.expander("Scatter Plot: MDVP:Fo(Hz) vs MDVP:Fhi(Hz)"):
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=data, x="MDVP:Fo(Hz)", y="MDVP:Fhi(Hz)", hue="status", palette="Set2")
    st.pyplot(fig3)
