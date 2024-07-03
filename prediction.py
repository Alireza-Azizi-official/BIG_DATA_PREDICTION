import numpy as np
import pandas as pd 
import pandas
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import PyPDF2
import tabula
import io 


# load, process, read the data file 
def l_p_data(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        dfs = []
        
        for page in range(len(pdf_reader.pages)):
            tables = tabula.read_pdf(file_path, pages = page + 1, multiple_tables = True)
            if tables: 
                dfs.extend(tables)
                
        if dfs:
            df = pd.concat(dfs, ignore_index = True)
        else:
            raise ValueError ('!! no tables found in the pdf')
        
        # Print column names and shape for debugging
        print("Columns:", df.columns)
        print("Shape:", df.shape)
        
        # Assuming the last column contains the values you want
        df['value'] = pd.to_numeric(df.iloc[:, -1].replace('[^\d.]', '', regex = True), errors = 'coerce')
        
        df = df.fillna(method = 'ffill') 
          
        return df[['value']] 

# feature extraction
def create_features(df):
    a = df['lag_1'] = df['value'].shift(1)
    df['rolling_mean_7'] = df['value'].rolling(window = 7).mean()
    return df.dropna()

# data spliting
def split_data(df, train_size = 0.8):
    train = df[:int(len(df)*train_size)]
    test = df[int(len(df)* train_size)]
    return train, test

# model training
def train_arima_model(train_data):
    model = ARIMA(train_data['value'], order= (1,1,1))
    result = model.fit()
    return result

# forecasting
def forecast(model, steps):
    forecast = model.forecast(steps = steps)
    return forecast

# evaluation
def evaluate_forecast(actual, predicted):
    rmse = sqrt(mean_squared_error(actual, predicted))
    return rmse

# main execution and run the project
if __name__  == '__main__':
    df = l_p_data('data.pdf')
    df = create_features(df)
    train, test = split_data(df)
    model = train_arima_model(train)
    forecast_horizon = 20
    prediction = forecast(model, steps = forecast_horizon)
    rmse = evaluate_forecast(test['value'][:forecast_horizon], prediction)
    print(f'RMSE: {rmse}')
    
    plt.figure(figsize=(12,6))
    plt.plot(test.index[:forecast_horizon], test['value'][:forecast_horizon], label = 'Actual')
    plt.plot(test.index[:forecast_horizon], prediction, label = 'Forecast')
    plt.title('time series forecast')
    plt.legend()
    plt.show()