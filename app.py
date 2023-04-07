## Importing libraries

import yfinance as yf
import Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import streamlit as st
import datetime
from datetime import date 
from plotly import graph_objects as go
import plotly.express as px
import time
from neuralprophet import set_random_seed 
import plotly.figure_factory as ff
import scipy
from scipy import stats
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import math
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from keras import layers
from chart_studio import plotly
import plotly.tools as tls
import plotly.offline as pyoff


## functions definitions 
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    #layout="wide"
)
 ###

st.title("Stock Closing price Predictor")
st.image('Images/main.gif')
st.caption('Picture Credit: https://insightimi.files.wordpress.com')
with st.sidebar:
    st.header('PAGE')
    page=st.selectbox('',
    ['EXPLORATION', 'MODELING'])
@st.cache_data
def load_data(ticker,a,b):
            data=yf.download(ticker,a,b)
            data.reset_index(inplace=True)
            #data=si.get_data(ticker)
            #data.reset_index(inplace=True)
            data=data[["Date","Open","Close","High","Low","Volume"]]
            #data=data.rename(columns={"index": "Date"})
            return data

#@st.cache_data
#def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
#    return df.to_csv().encode('utf-8')


def plot_line(df):
    fig = px.line(df, x='Date', y="Close")
    st.plotly_chart(fig,theme="streamlit",use_container_width=True)
#####
def plot_dist(df):
     data=[df.Close.values]
     group_labels=['Closing Price']
     fig = ff.create_distplot(data,group_labels)
     st.plotly_chart(fig,theme="streamlit",use_container_width=True)
#######
def plot_candle(df):
    fig = go.Figure(data=[go.Candlestick(x=df.Date,
                       open=df.Open, high=df.High,
                       low=df.Low, close=df.Close)])
    st.plotly_chart(fig,use_container_width=True) 
####
def plot_box(df):
     fig = px.box(df, y="Close")
     st.plotly_chart(fig,theme="streamlit",use_container_width=True)
#####


def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
        )
    )



## Data Exploration
end=datetime.date.today()
d = st.date_input(
    "Select The starting date of your data",value=datetime.date(2021, 1, 1),min_value=datetime.date(2016, 1, 1),max_value=end-datetime.timedelta(days=100))
start=d
stocks=("META","AMZN","AAPL","MSFT","TSLA","NFLX","GOOGL")
if page=="EXPLORATION":
    selected=st.selectbox("Select company for prediction",stocks)
    if st.button('Validate',key='Validate'):
        with st.spinner('Loading Data...'):
            Data=load_data(selected,start,end)
            my_bar=st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.001)
                my_bar.progress(percent_complete +1 ) 
        st.success('Data Loaded Successfully!', icon="✅")  

        ########## Raw data

        tab1, tab2= st.tabs(["🗃 Data","📈 Viz"])
        with tab1:
            st.header("Raw data")
            st.image(f"Images/{selected}.png")
        ### Last 5 Days
            st.subheader('Last 5 Days Data')
            st.write(Data.tail())
            st.write('Data has',len(Data),'Rows')
        ###### Stats Descriptive
            st.subheader('Descriptive stats')
            st.write(Analysis.analyse_num(Data))
            st.subheader('Statistics Test')
            shapiro_test = stats.shapiro(Data['Close'])
            st.markdown('Shapiro Test pvalue')
            st.write(round(shapiro_test.pvalue,4))
            st.caption('The Shapiro-Wilk test tests the null hypothesis \
                       that the data was drawn from a normal distribution.')
            st.markdown('Stationarity ADF test pvalue')
            st.write(round(adfuller(Data['Close'])[0],4))
            st.caption(' If the p-value is  less than α (0.05 in general), we reject the null hypothesis.\
                        This means the time series is stationary.\
                        In other words, it does not have time-dependent structure and constant variance over time.')
            #csv = convert_df(Data)
        #st.download_button(
         #   label="Download data as CSV",
          #  data=csv,
           # file_name=f'{selected}.csv',
            #)
        
        ####### Vizualisation
        with tab2:
            st.header('Visualization')
            st.image("Images/exploration.png")
        #####
            #plot_candle(Data)
            st.markdown('Closing Price over time ')
            plot_line(Data)
            col1, col2 = st.columns(2,gap="small")
            with col1:
                st.markdown('BoxPlot of Closing Price')
                plot_box(Data)
            with col2:
                st.markdown('Distribution of Closing Prices')
                plot_dist(Data)
            st.markdown('Candlesticks')
            plot_candle(Data)
        st.subheader('Decomposition') 
        decomposition = seasonal_decompose(Data['Close'], model='additive', period=12)
        fig = plot_seasonal_decompose(decomposition, dates=Data['Date'])
        st.plotly_chart(fig,use_container_width=True)
            #pyplot.show()
            # Code for decomposition


    ######################## COMPARISON Plot


    st.header(' Comparison Plots')
    st.warning('You can Only select up to 4 Stocks ')
    to_compare = st.multiselect(' Select Stocks to compare',
    stocks,max_selections=4)
    tab3, tab4= st.tabs(["🗃 Data","📈 Viz"])
    if len(to_compare)!=0:
        with tab3:
            st.subheader('Raw Data') 
            Datas=list()
            for i in range(len(to_compare)):
                Datas.append(load_data(to_compare[i],start,end))
            dates=Datas[0].Date
            final_Datas=pd.DataFrame()
            final_Datas['Date']=dates
            for i in range(len(Datas)):
                final_Datas[to_compare[i]]=Datas[i].Close 
            final_Datas.set_index('Date',inplace=True)
            st.write(final_Datas)
        with tab4:
             st.subheader('Closing Prices Overtime')
             st.line_chart(final_Datas)  
             hist_data = [a.Close for a in Datas]
             group_labels = to_compare
             fig = ff.create_distplot(hist_data, group_labels,curve_type='normal')
             st.subheader('Distribution Comparison of Closing Prices')
             st.plotly_chart(fig,use_container_width=True)
        #for i in Datas:
        #st.write(len(Datas))


########################### Page 2

### PREDICTIONS  
#selected=selected
else:
    Data = load_data(st.radio(
    "Pick the Data",stocks,horizontal=True),start,end)
    st.header('Predictions')
    n_days = st.slider('Future Days to Predict',min_value=7,max_value=30)
    model=st.radio("Select Your Model",('NeuralProphet','LSTM'),horizontal=True)
    if st.button('Predict'):
        if model=="NeuralProphet":
            model=NeuralProphet(daily_seasonality='auto',trend_reg_threshold=True,
                                num_hidden_layers=2,normalize='standardize')
            Data=Data[['Date','Close']].rename(columns={"Date": "ds", "Close":"y"})
            df_train, df_test = model.split_df(Data, valid_p=0.2,freq="D")  
            with st.spinner('Getting predictions Please Wait....'):
                set_random_seed(0)
                metrics=model.fit(df_train,freq='D',validation_df=df_test)
                #valid_metrics=model.test(df_test)  
            st.info('Predictions Completed')
            st.subheader('Model Insights')    
            futureplot= model.make_future_dataframe(Data,periods=n_days,n_historic_predictions=len(Data))
            forecast = model.predict(futureplot)
            st.markdown(' Metrics')
            st.write(metrics.tail(1))
            try :
                st.markdown('Model Parameters')
                fig_param = model.plot_parameters()
                st.plotly_chart(fig_param,use_container_width=True)
            except:
                st.error("An error occured while trying to print Model Parameters please pick\
                          a closer date from today and try again")
            st.subheader('Forecasting')
            st.markdown('forecast plot')
            fig1=model.plot(forecast)
            st.plotly_chart(fig1,use_container_width=True)
            st.markdown('Forecasted Data')
            st.write('Next ',n_days,' Days Predictions')
            st.write(forecast[['ds','yhat1']].rename(columns={'ds':'Date','yhat1':'Forecasted'}).tail(n_days))
        else :
             ######## LSTM
             data=Data.copy()
             Data.set_index('Date',inplace=True)
             Data=Data['Close']
             values = Data.values
             train_length=math.ceil(len(values)* 0.8)
             st.write('Training Data Length :',train_length)
             scaler = MinMaxScaler(feature_range=(0,1))
             scaled_data = scaler.fit_transform(values.reshape(-1,1))
             train_data = scaled_data[0:train_length,:]
             x_train = []
             y_train = []
             ##Create a 60-days window of historical prices (i-60) as our feature data (x_train)
             #and the following 60-days window as label data (y_train).
             for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
             x_train, y_train = np.array(x_train), np.array(y_train)
             x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
             ### Test data
             test_data = scaled_data[train_length-60:,:]
             x_test = []
             y_test = values[train_length:]
             for i in range(60, len(test_data)):
                            x_test.append(test_data[i-60:i, 0])
             x_test = np.array(x_test)
             x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
             model = keras.Sequential()
             model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
             model.add(layers.LSTM(100, return_sequences=False))
             model.add(layers.Dense(25))
             model.add(layers.Dense(1))
             st.subheader('Model Summary')
             model.compile(optimizer='adam', loss='mean_absolute_error')
             with st.spinner('Getting Predictions Please Wait...'):  
                  history=model.fit(x_train, y_train, batch_size= 1, epochs=10)
             st.subheader('Model Insights')
             st.markdown('Model Summary')
             model.summary(print_fn=lambda x: st.text(x))
             st.markdown('Loss history (MAE)')
             #st.write(history.history)
             fig,ax=plt.subplots(1,1)
             ax.plot(history.history['loss'])
             ax.set_title('model loss')
             ax.set_ylabel('loss')
             ax.set_xlabel('epoch')
             plotly_fig=tls.mpl_to_plotly(fig)
             st.plotly_chart(plotly_fig,use_container_width=True)
             predictions = model.predict(x_test)
             predictions = scaler.inverse_transform(predictions)
             rmse = np.sqrt(np.mean(predictions - y_test)**2)
             st.markdown('Metrics')
             st.write('RMSE',rmse)
             st.subheader('Forecast')
             f_data=data
             data=data.set_index('Date')[['Close']]
             train = data[:train_length]
             validation = data[train_length:]
             validation['Predictions'] = predictions
             st.markdown('Forecast Plot')
             fig,ax=plt.subplots(1)
             ax.set_title('Model')
             ax.set_xlabel('Date')
             ax.set_ylabel('Close Price USD ($)')
             ax.plot(train)
             ax.plot(validation[['Close', 'Predictions']])
             ax.legend(['Train', 'Val', 'Predictions'], loc='upper right')
             plotly_fig=tls.mpl_to_plotly(fig)
             #pyoff.plot(plotly_fig)
             st.plotly_chart(plotly_fig,use_container_width=True)