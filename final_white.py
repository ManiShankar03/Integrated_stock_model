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

# Load the additional datasets for sentiment analysis
news_data = pd.read_csv('news_data.csv')
stock_data = pd.read_csv('new_historical_stock_data_yahoo.csv')

# Convert date columns to datetime
news_data['published_utc'] = pd.to_datetime(news_data['published_utc'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.set_index('Date')

# Set the directory where models are saved
model_save_path = 'saved_models'

# Streamlit app layout
st.title('Integrated Model of Stock Price Prediction Using Sentiment analysis on News and Financial Ratios')

# List of companies for the dropdown, with Google (GOOGL) as the first option
companies = df['symbol'].unique()
companies = sorted(companies)
if 'GOOGL' in companies:
    companies = ['GOOGL'] + [comp for comp in companies if comp != 'GOOGL']

# Company selection dropdown (Google as the default)
company_symbol = st.selectbox('Select Company', companies)

# Date range selection for sentiment analysis and stock data
selected_date = st.date_input(
    "Select a Date for Sentiment and Stock Data (range: 07/07/2024 to 06/08/2024)",
    min_value=pd.to_datetime('2024-07-07'),
    max_value=pd.to_datetime('2024-08-06')
)

# Function to analyze sentiments
def analyze_sentiments(selected_date, selected_company):
    selected_date = pd.to_datetime(selected_date)
    filtered_news = news_data[(news_data['published_utc'].dt.date == selected_date.date()) & (news_data['ticker'] == selected_company)]
    sentiment_counts = filtered_news['sentiment_given'].value_counts()

    if sentiment_counts.empty:
        return 'No news available for the selected date and company', sentiment_counts
    most_common_sentiment = sentiment_counts.idxmax()
    return most_common_sentiment, sentiment_counts

# Display sentiment analysis result
if st.button('Analyze Sentiments'):
    sentiment_result, sentiment_counts = analyze_sentiments(selected_date, company_symbol)
    st.write(f"Most Common Sentiment: {sentiment_result}")
    st.write(f"Sentiment Counts:\n{sentiment_counts}")

# Function to plot stock prices for the next three days
def plot_next_three_days(selected_date, selected_company):
    selected_date = pd.to_datetime(selected_date)
    company_data = stock_data[stock_data['ticker'] == selected_company]

    if company_data.empty:
        st.write(f"No data available for company {selected_company}.")
        return

    next_three_days = pd.date_range(start=selected_date, periods=4)
    missing_dates = [date for date in next_three_days if date not in company_data.index]

    if missing_dates:
        st.write(f"Data missing for the following dates: {missing_dates}")
        return

    next_three_days_data = company_data.loc[next_three_days]
    
    # Set white background
    plt.figure(figsize=(10, 5), facecolor='white')  # Set figure background to white
    ax = plt.gca()  # Get current axes

    # Set the plot (axes) background to white
    ax.set_facecolor('white')

    # Plot the data with a black line
    plt.plot(next_three_days_data.index, next_three_days_data['Close'], marker='o', linestyle='-', color='black')  

    # Customize title, labels, and grid
    plt.title(f'Stock Prices for {selected_company} for the Next Three Days Starting {selected_date.date()}', color='black')
    plt.xlabel('Date', color='black')
    plt.ylabel('Stock Price (Close)', color='black')

    # Customize ticks and grid for white background
    plt.grid(True, color='gray')
    plt.xticks(rotation=45, color='black')
    plt.yticks(color='black')

    # Tight layout to avoid cutting off labels
    plt.tight_layout()

    # Show the plot using Streamlit
    st.pyplot(plt)

# Button to plot stock prices for the next three days
if st.button('Plot Stock Prices for Next Three Days'):
    plot_next_three_days(selected_date, company_symbol)

# Display the first few rows of the stock data
company_data = df[df['symbol'] == company_symbol]
st.write("Company Sample Stock Data", company_data.head())

# Prepare the data for LSTM model
data = company_data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Load the corresponding model for the selected company
model_filename = os.path.join(model_save_path, f"{company_symbol}_lstm_model.h5")
model = load_model(model_filename)

# Create dataset function
def create_dataset(dataset, time_step=4):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Train and test data split
time_step = 4
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted stock prices with white background
fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
ax.set_facecolor('white')
ax.plot(data.index[:len(train_predict)], y_train, label='Actual Train', color='blue')
ax.plot(data.index[len(train_predict):(len(train_predict) + len(test_predict))], y_test, label='Actual Test', color='green')
ax.plot(data.index[:len(train_predict)], train_predict, label='Predicted Train', color='red')
ax.plot(data.index[len(train_predict):(len(train_predict) + len(test_predict))], test_predict, label='Predicted Test', color='orange')
ax.set_title(f'Actual vs. Predicted Stock Prices for {company_symbol}', color='black')
ax.set_xlabel('Date', color='black')
ax.set_ylabel('Stock Price', color='black')

# Customize ticks and labels for white background
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
legend = ax.legend(facecolor='white', edgecolor='black')
for text in legend.get_texts():
        text.set_color('black') 
ax.grid(True, color='gray')

st.pyplot(fig)

# Forecast for the next two quarters
last_4_quarters = scaled_data[-time_step:]
next_input = last_4_quarters.reshape(1, time_step, 1)

future_predictions = []
for _ in range(2):
    future_price = model.predict(next_input)
    future_predictions.append(future_price[0, 0])
    future_price = future_price.reshape(1, 1, 1)
    next_input = np.append(next_input[:, 1:, :], future_price, axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot future predictions with white background
st.write("Forecasted Stock Prices for the Next Two Quarters")
quarters_ahead = ['Q1', 'Q2']
fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor='white')
ax2.set_facecolor('white')
ax2.bar(quarters_ahead, future_predictions.flatten(), color='blue')
ax2.set_title(f'Forecasted Stock Prices for {company_symbol}', color='black')
ax2.set_xlabel('Quarter Ahead', color='black')
ax2.set_ylabel('Stock Price', color='black')

ax2.tick_params(axis='x', colors='black')
ax2.tick_params(axis='y', colors='black')

st.pyplot(fig2)