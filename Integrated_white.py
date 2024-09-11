import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# Load the data (replace with actual path to your data file)
file_path = 'final_csv_stock_financial.csv'
df = pd.read_csv(file_path)

# Set the directory where models are saved
model_save_path = 'saved_models'

# Streamlit app layout
st.title('Integrated Model of Stock Price Prediction Using Financial Ratios')

# List of companies for the dropdown, with Google (GOOGL) as the first option
companies = df['symbol'].unique()
companies = sorted(companies)
if 'GOOGL' in companies:
    companies = ['GOOGL'] + [comp for comp in companies if comp != 'GOOGL']

# Company selection dropdown (Google as the default)
company_symbol = st.selectbox('Select Company', companies)

# Filter data for the selected company
company_data = df[df['symbol'] == company_symbol]

# Display the first few rows of the data
st.write("Company Sample Stock Data", company_data.head())

# Prepare the data
data = company_data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Load the corresponding model for the selected company
model_filename = os.path.join(model_save_path, f"{company_symbol}_lstm_model.h5")
model = load_model(model_filename)

# Define a function to create the dataset
def create_dataset(dataset, time_step=4):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Create train and test datasets
time_step = 4  # 4 quarters (1 year window)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the data for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions and actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Set plot style for white background
plt.style.use('default')

# Plot actual vs predicted stock prices
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data.index[:len(train_predict)], y_train, label='Actual Train', color='blue')
ax.plot(data.index[len(train_predict):(len(train_predict) + len(test_predict))], y_test, label='Actual Test', color='green')
ax.plot(data.index[:len(train_predict)], train_predict, label='Predicted Train', color='red')
ax.plot(data.index[len(train_predict):(len(train_predict) + len(test_predict))], test_predict, label='Predicted Test', color='orange')
ax.set_title(f'Actual vs. Predicted Stock Prices for {company_symbol}', color='black')
ax.set_xlabel('Date', color='black')
ax.set_ylabel('Stock Price', color='black')

# Customize ticks and labels to be visible on white background
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

ax.legend(facecolor='white', edgecolor='black')
ax.grid(True, color='gray')

# Display the plot
st.pyplot(fig)

# Forecast the next two quarters
last_4_quarters = scaled_data[-time_step:]  # Get the last 4 quarters (1 year of data)
next_input = last_4_quarters.reshape(1, time_step, 1)  # Reshape to (1, 4, 1) for input to LSTM

future_predictions = []
for _ in range(2):  # Predicting 2 quarters ahead
    future_price = model.predict(next_input)
    future_predictions.append(future_price[0, 0])  # Append the predicted price
    future_price = future_price.reshape(1, 1, 1)
    next_input = np.append(next_input[:, 1:, :], future_price, axis=1)  # Sliding window

# Inverse transform the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot the future predictions for the next two quarters
st.write("Forecasted Stock Prices for the Next Two Quarters")
quarters_ahead = ['Q1', 'Q2']
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(quarters_ahead, future_predictions.flatten(), color='orange')
ax2.set_title(f'Forecasted Stock Prices for {company_symbol}', color='black')
ax2.set_xlabel('Quarter Ahead', color='black')
ax2.set_ylabel('Stock Price', color='black')

# Customize ticks and labels to be visible on white background
ax2.tick_params(axis='x', colors='black')
ax2.tick_params(axis='y', colors='black')

# Display the future predictions plot
st.pyplot(fig2)