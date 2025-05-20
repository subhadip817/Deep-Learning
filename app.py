
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import streamlit as st

model=keras.models.load_model('model.h5')

## Load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit app
st.title('Customer Churn Preiction')


tenure=st.number_input('Tenure',0,10)
geo=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,100)
creditscore=st.slider('Credit Score',0,900)
extimated_salary=st.number_input('Estimated Salary')
balance=st.number_input('Balance')
has_credit_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
no_of_products=st.slider('No of Product',0,10)

input_data=pd.DataFrame({
    'CreditScore':[creditscore],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_products],
    'HasCrCard':[has_credit_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[extimated_salary]
})

geo_encoded=onehot_encoder_geo.transform([[geo]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([geo_encoded_df,input_data],axis=1)

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob>0.02:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
