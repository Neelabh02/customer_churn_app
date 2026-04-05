import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('churn_model.h5')

# Load encoders
with open('le_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('ohe.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# UI
st.title("Customer Churn Prediction")

#  INPUTS 
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
credit_score = st.number_input('Credit Score')
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

#  DATAFRAME 
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# ONE HOT ENCODING 
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

# Combine
input_df = pd.concat([input_df, geo_df], axis=1)

# SCALING 
scaled_input = scaler.transform(input_df)

#  PREDICTION 
prediction = model.predict(scaled_input)
prob = prediction[0][0]

if prob > 0.5:
    st.write(f"Customer likely to churn: {prob*100:.2f}%")
else:
    st.write(f"Customer not likely to churn: {(1-prob)*100:.2f}%")

