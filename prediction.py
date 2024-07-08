import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import PyPDF2
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to load, process, and read the data file
def load_process_data(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Use regex to find numeric values
        numbers = re.findall(r'\d+\.?\d*', text)
        df = pd.DataFrame({'value': [float(num) for num in numbers]})
    return df

# Function to split data
def split_data(df, train_size=0.8):
    train = df[:int(len(df)*train_size)]
    test = df[int(len(df)*train_size):]
    return train, test

# Function to train the ARIMA model
def train_arima_model(train_data):
    model = ARIMA(train_data['value'], order=(1,1,1))  
    result = model.fit()
    return result

# Function to forecast
def forecast(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Function to evaluate the forecast
def evaluate_forecast(actual, predicted):
    rmse = sqrt(mean_squared_error(actual, predicted))
    return rmse

# Main execution
if __name__ == '__main__':
    df = load_process_data('data.pdf')
    train, test = split_data(df)
    model = train_arima_model(train)

    # Update the forecast horizon to predict 10 to 20 levels
    forecast_horizon = 20  # Adjust this value between 10 and 20
    prediction = forecast(model, steps=forecast_horizon)
    rmse = evaluate_forecast(test['value'][:forecast_horizon], prediction)
    print(f'RMSE: {rmse}')

    plt.figure(figsize=(12,6))
    plt.plot(test.index[:forecast_horizon], test['value'][:forecast_horizon], label='Actual')
    plt.plot(test.index[:forecast_horizon], prediction, label='Forecast')
    plt.title('Time Series Forecast')
    plt.legend()
    plt.show()
