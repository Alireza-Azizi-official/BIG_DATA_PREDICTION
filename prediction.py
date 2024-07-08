import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import PyPDF2
import re

# load, process, read the data file
def l_p_data(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Use regex to find numeric values
        numbers = re.findall(r'\d+\.?\d*', text)
        df = pd.DataFrame({'value': [float(num) for num in numbers]})
        return df

# feature extraction 
def create_features(df):
  df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
  df['lag_1'] = df['value'].shift(1)
  return df.dropna()

# data splitting
def split_data(df, train_size=0.8):
    train = df[:int(len(df)*train_size)]
    test = df[int(len(df)*train_size):]
    return train, test

# model training
def train_arima_model(train_data):
    model = ARIMA(train_data['value'], order=(1,1,1))
    result = model.fit()
    return result

# forecasting
def forecast(model, steps):
    forecast = model.forecast(steps=forecast_horizon)
    return forecast

# evaluation
def evaluate_forecast(actual, predicted):
    rmse = sqrt(mean_squared_error(actual, predicted))
    return rmse

# main execution and run the project
if __name__ == '__main__':
    df = l_p_data('data.pdf')
    df = create_features(df)
    train, test = split_data(df)
    model = train_arima_model(train)
    forecast_horizon = 20
    prediction = forecast(model, steps=forecast_horizon)
    rmse = evaluate_forecast(test['value'][:forecast_horizon], prediction)
    print(f'RMSE: {rmse}')
    
    plt.figure(figsize=(12,6))
    plt.plot(test.index[:forecast_horizon], test['value'][:forecast_horizon], label='Actual')
    plt.plot(test.index[:forecast_horizon], prediction, label='Forecast')
    plt.title('Time Series Forecast')
    plt.legend()
    plt.show()