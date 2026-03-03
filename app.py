import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("🩺 Medical Insurance Cost Predictor")

# 2. Load and Prepare Data (Cached for performance)
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df = load_data()

# 3. Preprocessing (Training the model internally)
# We need to replicate the encoding used in your notebook for the model to understand inputs
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])

X = df.drop(['charges'], axis=1)
y = df['charges']

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# 4. User Input Interface
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# Mapping inputs to match the model's expected integer encoding
# We need to ensure these mappings match how LabelEncoder transformed the data
sex_map = {'male': 1, 'female': 0} # Adjust based on your specific LabelEncoder output
smoker_map = {'yes': 1, 'no': 0}
region_map = {'southeast': 2, 'southwest': 3, 'northeast': 0, 'northwest': 1}

# 5. Prediction Logic
if st.button("Predict Insurance Cost"):
    input_data = np.array([[
        age, 
        sex_map[sex], 
        bmi, 
        children, 
        smoker_map[smoker], 
        region_map[region]
    ]])
    
    prediction = model.predict(input_data)
    
    st.success(f"The estimated insurance cost is **${prediction[0]:,.2f}**")

# 6. Optional: Show Data Overview
if st.checkbox("Show Raw Data"):
    st.write(df.head())
