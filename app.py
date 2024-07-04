import streamlit as st
import pandas as pd
import numpy as np
import pickle 

# LOAD THE MODEL 
clf = pickle.load(open("mymodel.pkl","rb"))

def predict(data):
    clf = pickle.load(open("mymodel.pkl","rb"))
    return clf.predict(data)

st.title("Advertising Spends Prediction Using Machine Learning")
st.markdown("This Model Identify Total Spends in advertising")

st.header("Advertising Spend on Various Media")
col1,col2 = st.columns(2)

with col1:
    st.text("TV")
    tv = st.slider("Adver. Spends on TV",1.0,10000.0,0.5)
    st.text("Radio")
    radio = st.slider("Adver. Spends on Radio",1.0,10000.0,0.5)
    st.text("Newspaper")
    newspaper = st.slider("Adver. Spends on Newspaper",1.0,10000.0,0.5)

st.text('')
if st.button("Sales Prediction"):
    result = clf.predict(np.array([[tv,radio,newspaper]]))
    st.text(result[0])

st.markdown("Developed By Nishant Sawaimoon at NIELT Daman")