# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
import pickle
# load the saved model
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def xgb_shap_transform_scale(original_shap_values, Y_pred, which):    
    from scipy.special import expit    
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    from scipy.special import expit 
    #Importing the logit function for the base value transformation    
    untransformed_base_value = original_shap_values.base_values[-1]    
    #Computing the original_explanation_distance to construct the distance_coefficient later on    
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]    
    base_value = expit(untransformed_base_value) 
    # = 1 / (1+ np.exp(-untransformed_base_value))    
    #Computing the distance between the model_prediction and the transformed base_value    
    distance_to_explain = Y_pred[which] - base_value    
    #The distance_coefficient is the ratio between both distances which will be used later on    
    distance_coefficient = original_explanation_distance / distance_to_explain    
    #Transforming the original shapley values to the new scale    
    shap_values_transformed = original_shap_values / distance_coefficient    
    #Finally resetting the base_value as it does not need to be transformed    
    shap_values_transformed.base_values = base_value    
    shap_values_transformed.data = original_shap_values.data    
    #Now returning the transformed array    
    return shap_values_transformed


plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time Fraud Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>FIB-4 计算器</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'> </h1>", unsafe_allow_html=True)


# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('年龄 (岁)', 0.0, 100.0, 0.0)
    a2 = st.sidebar.slider('AST (U/L)', 0.0, 1000.0, 0.0)
    a3 = st.sidebar.slider('PLT (×109/L)', 0.0, 1000.0, 0.0)
    a4 = st.sidebar.slider('ALT (U/L)', 0.0, 1000.0, 0.0)

    
    output = [a1,a2,a3,a4]
    return output

outputdf = user_input_features()

#st.header('👉 Make predictions in real time')
colnames = ['年龄 (岁)','AST (U/L)','PLT (×109/L)','ALT (U/L)']
outputdf = pd.DataFrame([outputdf], columns= colnames)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)


p1 = (outputdf.iat[0,0]*outputdf.iat[0,1])/(outputdf.iat[0,2]*np.sqrt(outputdf.iat[0,3]))
#modify output dataframe

placeholder6 = st.empty()
with placeholder6.container():
    st.subheader('Part1: User input parameters below ⬇️')
    st.write(outputdf)


placeholder7 = st.empty()
with placeholder7.container():
    st.subheader('Part2: Output results ⬇️')
    st.write(f'FIB-4 = {p1}')

placeholder8 = st.empty()
with placeholder8.container():   
    st.subheader('Part3: Formulation ⬇️')
    st.write('FIB-4 = [年龄 (岁) ×AST (U/L)] ÷ [PLT (×109/L) ×ALT (U/L)的平方根]')

