import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set up page
st.set_page_config(page_title="Insurance Insights", layout="wide")
st.title("📊 Health Insurance Analysis & Prediction")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df = load_data()

# Tabs
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Cost Predictor"])

with tab1:
    st.header("Data Insights")
    
    # Heatmap
    st.subheader("Correlation Matrix")
    # We create a temporary DF for correlation to handle encoding only for the plot
    temp_df = df.copy()
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        temp_df[col] = le.fit_transform(temp_df[col])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Row for sub-plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Smoker Distribution")
        fig2, ax2 = plt.subplots()
        df['smoker'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, startangle=90)
        st.pyplot(fig2)

    with col2:
        st.subheader("Charges by Region")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='region', y='charges', data=df, ax=ax3)
        st.pyplot(fig3)

with tab2:
    st.header("Predict Your Insurance Cost")
    # ... (Your existing input logic from the previous step)
    # Ensure the inputs match your model encoding!
