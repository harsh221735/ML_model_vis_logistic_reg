import time
from networkx import display
import requests
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate 2D binary classification data

if "trained_data" not in st.session_state:
    st.session_state.trained_data = None
back_url = "http://localhost:5001"  

response1 = requests.get(back_url + "/display")
data_json = response1.json()
df = pd.DataFrame(data_json["data"])
st.dataframe(df, width= 'stretch')

st.set_page_config(page_title="Logistic Regression Visualizer", layout="centered")
st.title("Logistic Regression with Gradient Descent")
mode = st.selectbox('Enter gradient descent mode', ('batch', 'mini-batch', 'stochastic'))
    
    # Execute the POST request
if mode == 'batch':
    lr = st.number_input('Enter learning rate: ', value=0.01)
    epochs = st.number_input('Enter number of epochs: ', value=10)
    batch_size = None
elif mode == 'mini-batch':
    lr = st.number_input('Enter learning rate: ', value=0.01)
    epochs = st.number_input('Enter number of epochs: ', value=10)
    batch_size = st.number_input('Enter batch size: ', value=5)
elif mode == 'stochastic':
    lr = st.number_input('Enter learning rate: ', value=0.01)
    epochs = st.number_input('Enter number of epochs: ', value=10)
    batch_size = None
payload = {"mode": mode,
           "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size
           }
#slider control
slider_val = st.slider("Adjust the speed of visualization", min_value=0.01, max_value=0.1, value=0.1, step=0.01)
if st.button("Train Model"):
    response = requests.post(back_url + "/train", json=payload)
    data = response.json()
    if response.status_code == 200:
        st.success(data["message"])
        st.write("Coefficients dekh :", data["coefficients"])
        slopes_array = data["slopes_array"]
    y_pred = data['y_pred']
    new_y = []
    for i in y_pred:
        if i < 0.5:
            new_y.append(0)
        elif i > 0.5:
            new_y.append(1)
    #visualization
    st.session_state.trained_data = response.json()

# 3. Check if model is trained before showing Visualize button
if st.session_state.trained_data:
    data = st.session_state.trained_data
    if st.button("Visualize karne ke liye mujhe dabao 💦 ̣"):
        X = np.linspace(
            df["coordinate X"].min(),
            df["coordinate X"].max(),
            100
        )
        X = X.reshape(len(X),1)
        plot_placeholder = st.empty()
        count = 1
        batch_size = data['batch_size']
        slopes_array = data['slopes_array']
        for j in range(epochs * batch_size):
            #print(count)
            if j%batch_size == 0 :
                count = count +1
            i = slopes_array[j-1]
            fig,ax = plt.subplots()
            ax.scatter(df["coordinate X"], df["coordinate Y"])
            Y = -(np.dot(X,i[2]) + i[0])/i[1]
            ax.plot(X,Y,color = 'red')
            ax.set_xlabel(f'epoch {count - 1} , iterations {j%batch_size}')
            plot_placeholder.pyplot(fig)
        #epoch calculation
            plt.close(fig) # Important: Clear memory to avoid memory leaks
            time.sleep(slider_val)