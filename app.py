'''import streamlit as st
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

    '''
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")

# 1. Load Data & Train Model
@st.cache_data
def get_data():
    df = pd.read_csv('insurance.csv')
    return df

@st.cache_resource
def train_model(df):
    df_model = df.copy()
    le = LabelEncoder()
    # Apply encoding
    for col in ['sex', 'smoker', 'region']:
        df_model[col] = le.fit_transform(df_model[col])
    
    X = df_model.drop(columns='charges')
    y = df_model['charges']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le

df = get_data()
model, le = train_model(df)

# Sidebar: Inputs
st.sidebar.header("Patient Parameters")
age = st.sidebar.number_input("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.number_input("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

# Mapping for the model
# We map inputs to match LabelEncoder logic
# IMPORTANT: This order must match your training data columns!
# Assuming the order is: age, sex, bmi, children, smoker, region
def get_encoded_value(val, col_name):
    # This is a robust way to match the encoder
    mapping = {
        'sex': {'male': 1, 'female': 0},
        'smoker': {'yes': 1, 'no': 0},
        'region': {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    }
    return mapping[col_name].get(val, 0)

# Main Application
st.title("🩺 Medical Insurance Cost Predictor")
tabs = st.tabs(["📊 Exploratory Data Analysis", "🔮 Cost Predictor"])

with tabs[0]:
    st.subheader("Visual Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Correlation Matrix**")
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.write("**Charges by Region**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='region', y='charges', data=df, ax=ax2)
        st.pyplot(fig2)

with tabs[1]:
    st.subheader("Predictive Modeling")
    if st.button("Calculate Insurance Cost"):
        input_data = np.array([[
            age, 
            get_encoded_value(sex, 'sex'),
            bmi,
            children,
            get_encoded_value(smoker, 'smoker'),
            get_encoded_value(region, 'region')
        ]])
        
        prediction = model.predict(input_data)
        st.metric(label="Estimated Insurance Cost", value=f"${prediction[0]:,.2f}")
