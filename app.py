import streamlit as st
import pandas as pd
import pickle

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

st.set_page_config(page_title="Insurance Risk AI", page_icon="🛡️")
st.title("🛡️ Insurance Claim Risk Predictor")
st.write("This AI predicts if a patient will have **High** vs **Low** insurance charges.")

st.sidebar.header("Patient Details")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
sex = st.sidebar.radio("Sex", ["male", "female"])
smoker = st.sidebar.radio("Smoker?", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Analyze Risk Profile"):
    data = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [1 if sex == 'male' else 0],
        'smoker': [1 if smoker == 'yes' else 0]
    }
    
    data['region_northwest'] = [1 if region == 'northwest' else 0]
    data['region_southeast'] = [1 if region == 'southeast' else 0]
    data['region_southwest'] = [1 if region == 'southwest' else 0]
    
    input_df = pd.DataFrame(data)
    input_df = input_df[model_features]

    prediction = model.predict(input_df)
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: 🚩 High Claim Risk")
        st.write("This patient is predicted to exceed the median insurance cost.")
    else:
        st.success("### Result: ✅ Low Claim Risk")
        st.write("This patient is predicted to be below the median insurance cost.")