
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import streamlit as st
from datetime import date 
from plotly import graph_objects as go
import time
import seaborn as sns
from neuralprophet import set_random_seed 


st.title("       Stock Closing price Predictor      ")
stocks=("FB","AMZN","QCOM")
selected=st.selectbox("Select company for prediction",stocks)

st.warning('Please select a number.')
n_days=st.slider("Days of predictions:", 0 , 365 )
period=n_days
time.sleep(8)
st.success('Days for forecasting selected')
start='2018-01-01'
end=date.today().strftime("%Y-%m-%d")

@st.cache
def load_data(ticker,a,b):
    data=yf.download(ticker,a,b)
    data.reset_index(inplace=True)
    #data=si.get_data(ticker)
    #data.reset_index(inplace=True)
    data=data[["Date","Open","Close","High","Low","Volume"]]
    #data=data.rename(columns={"index": "Date"})
    return data

with st.spinner('Loading Data...'):
    time.sleep(4)
Data=load_data(selected,start,end) 

my_bar=st.progress(0)
for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete +1 )    
          
with st.spinner('Data successfully loaded !'):
    time.sleep(3)
  
st.subheader("Raw data")
st.write(Data.tail())
st.write('Data has',len(Data),'Rows')
st.header('Vizualization')

def plot_close(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Close']))
    fig.layout.update(title_text="Closing price over time",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
def plot_candle(df):    
    fig=go.Figure()
    fig =fig.add_trace(go.Candlestick(x=df['Date'].dt.year,
                       open=df['Open'], high=df['High'],
                       low=df['Low'], close=df['Close']))
    fig.layout.update(title_text="Stock Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    
plot_candle(Data)
plot_close(Data)



st.header('Predictions')
    
model=NeuralProphet()

Data2=Data[['Date','Close']]
Data2=Data2.rename(columns={"Date": "ds", "Close":"y"})   
with st.spinner('Getting predictions Please Wait....'):
    time.sleep(10)

@st.cache
def Predictor(df):
    set_random_seed(0)
    model.fit(df,validate_each_epoch=True,valid_p=0.2,freq='D')

Predictor(Data2)

with st.spinner('Predictions Completed'):
    time.sleep(3)

st.subheader('Prediction Results')    

futureplot= model.make_future_dataframe(Data2,periods=n_days,n_historic_predictions=len(Data2))
forecast = model.predict(futureplot)
fig1=model.plot(forecast)
st.plotly_chart(fig1)


st.write('Next ',n_days,' Days Predictions')
st.write(forecast[['ds','yhat1']].tail(n_days))



#@st.cache
#def future(x,mod):
 ##  return future_x

#futuredf=future(n_days,model)
#st.write(futuredf.tail(n_days))

 
