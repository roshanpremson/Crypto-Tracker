import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from prophet.plot import plot_plotly

# App title
st.title("Real-Time Crypto Price Tracker + Forecast")
st.write("Track and forecast cryptocurrency prices with this interactive tool.")

# User configuration
st.subheader("⚙️ Configuration")
coin = st.selectbox("Choose a cryptocurrency", ["bitcoin", "ethereum", "cardano", "dogecoin"])
days = st.slider("Days of historical data", 30, 180, 90)
forecast_horizon = st.slider("Forecast for next _ days", 7, 30, 14)

# Fetch historical data
url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
r = requests.get(url)

if r.status_code == 200:
    data = r.json()

    # Validate 'prices' in response
    if 'prices' in data:
        # Process price data
        prices = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
        prices['ds'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices['y'] = prices['price']
        prices = prices[['ds', 'y']]
    else:
        st.error("API response does not contain 'prices' data.")
        st.stop()

    # Market Cap Data
    if 'market_caps' in data and data['market_caps']:
        market_caps = pd.DataFrame(data['market_caps'], columns=["timestamp", "market_cap"])
        market_caps['ds'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
        market_caps = market_caps[['ds', 'market_cap']]
    else:
        market_caps = pd.DataFrame()

    # Total Volume Data
    if 'total_volumes' in data and data['total_volumes']:
        total_volumes = pd.DataFrame(data['total_volumes'], columns=["timestamp", "total_volume"])
        total_volumes['ds'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')
        total_volumes = total_volumes[['ds', 'total_volume']]
    else:
        total_volumes = pd.DataFrame()

else:
    st.error("Failed to fetch data from CoinGecko API.")
    st.stop()

# Technical Indicator
prices['SMA_20'] = prices['y'].rolling(window=20).mean()

# Display Key Metrics
st.subheader("Key Metrics")
col1, col2 = st.columns(2)

if not market_caps.empty:
    latest_market_cap = market_caps['market_cap'].iloc[-1]
    col1.metric("Market Cap", f"${latest_market_cap:,.2f}")

if not total_volumes.empty:
    latest_total_volume = total_volumes['total_volume'].iloc[-1]
    col2.metric("24h Volume", f"${latest_total_volume:,.2f}")

# Price Chart
st.subheader(f"{coin.capitalize()} Price Chart (Last {days} Days) with 20-Day SMA")
st.line_chart(prices.set_index("ds")[['y', 'SMA_20']])

# Additional Charts
st.subheader("Additional Data")

if not market_caps.empty:
    st.write(f"{coin.capitalize()} Market Cap")
    st.line_chart(market_caps.set_index("ds")["market_cap"])

if not total_volumes.empty:
    st.write(f"{coin.capitalize()} Total Volume")
    st.line_chart(total_volumes.set_index("ds")["total_volume"])

# Forecast
st.subheader(f"Forecast: Next {forecast_horizon} Days")
st.write("Forecast generated using Facebook Prophet.")

model = Prophet()
model.fit(prices)
future = model.make_future_dataframe(periods=forecast_horizon)
forecast = model.predict(future)

fig = plot_plotly(model, forecast)
st.plotly_chart(fig)
