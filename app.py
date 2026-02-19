import joblib
import streamlit as st
import numpy as np
import pandas as pd
from utils import columns, PreProcessor

model = joblib.load("model2.joblib")
st.title("will you survived if you were amont titanic passenger")
passengerid = st.text_input("Input Passenger ID", "33")
pclass = st.selectbox("Choose Class", [1, 2, 3])
name = st.text_input("Enter Your Name", "Joen Doe")
sex = st.select_slider("Choose Sex", ["male", "female"])
age = st.slider("Choose Age", 0, 100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,10)
ticket = st.text_input("Input Ticket Number", "8585") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

def predict():
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked])
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.success("Passenger Survived :thumbsup:")
    else:
        st.error("Passenger did not survived :thumbsdown:")

trigger = st.button("predict", on_click=predict)