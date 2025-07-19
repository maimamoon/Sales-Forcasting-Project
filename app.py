import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load the trained model
model = load_model("model.keras")

# Load and preprocess sales data
@st.cache_data
def load_data():
    df = pd.read_csv("train 2.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    return daily_sales

daily_sales = load_data()

# Normalize the data
scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(daily_sales[['sales']])
window_size = 30

st.title("📈 Store Sales Forecasting using LSTM")

forecast_days = st.slider("Select number of days to forecast", 1, 60, 30)

# Forecast
input_seq = scaled_sales[-window_size:].reshape(1, window_size, 1)
forecast = []

for _ in range(forecast_days):
    pred = model.predict(input_seq)[0][0]
    forecast.append(pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Create future date range
last_date = daily_sales['date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# Display results
forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecasted_sales': forecast_inv.flatten()
})

st.subheader("📅 Forecasted Sales")
st.dataframe(forecast_df)

# Plot
st.subheader("📊 Sales Forecast Chart")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_sales['date'], daily_sales['sales'], label='Historical Sales')
ax.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='Forecast', color='green')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)
