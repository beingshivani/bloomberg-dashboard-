# ------------------ IMPORTS ------------------
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image  # Pillow for image validation
from typing import List, Tuple

# ------------------ CONFIG ------------------
WATCHLIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]  # NSE tickers
NEWS_API_KEY = "your_newsapi_key_here"  # Replace with your NewsAPI.org key
BIG_MOVE_THRESHOLD = 1.0  # % change for alerts

# ------------------ FUNCTIONS ------------------
def get_intraday_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="5m")
    if data.empty:
        return pd.DataFrame()

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['AveragePrice'] = data['Close'].expanding().mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def get_top_movers(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame()
    for t in tickers:
        stock = yf.Ticker(t)
        hist = stock.history(period="1d")
        if not hist.empty:
            df.loc[t, 'Change%'] = (
                (hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1] * 100
            )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    gainers = df.sort_values('Change%', ascending=False)
    losers = df.sort_values('Change%', ascending=True)
    return gainers, losers


def get_news(ticker: str, keywords: List[str] = []) -> List[Tuple[str, str, str]]:
    query = ticker
    if keywords:
        query += " " + " OR ".join(keywords)
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        articles = response.json().get('articles', [])[:5]
        return [(a['title'], a['source']['name'], a['publishedAt']) for a in articles]
    except requests.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []


def get_events(ticker: str) -> List[str]:
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        events = [f"{idx}: {cal.loc[idx][0]}" for idx in cal.index]
        return events
    except Exception as e:
        st.error(f"Error fetching events: {e}")
        return []

def is_valid_image(uploaded_file) -> bool:
    try:
        img = Image.open(uploaded_file)
        img.verify()  # Verify that it is an image
        return True
    except Exception:
        return False

# ------------------ STREAMLIT DASHBOARD ------------------
st.set_page_config(page_title="Mini Bloomberg Lite", layout="wide")
st.title("📈 Mini Bloomberg Lite - Live Terminal")

# Optional Image Upload
st.sidebar.header("📷 Upload Company Logo or Chart")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    if is_valid_image(uploaded_file):
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    else:
        st.sidebar.error("Invalid image file.")

# Select Ticker
selected_ticker = st.selectbox("Select Ticker", WATCHLIST)

# Intraday Data & Technicals
data = get_intraday_data(selected_ticker)
if not data.empty:
    st.subheader(f"Intraday Price & Technicals - {selected_ticker}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50'))
    fig.add_trace(go.Scatter(x=data.index, y=data['AveragePrice'], mode='lines',
                             name='AvgPrice', line=dict(dash='dot')))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data[['Close', 'MA20', 'MA50', 'AveragePrice', 'RSI']].tail(10))
else:
    st.warning("No intraday data available right now.")

# Top Movers
st.subheader("📊 Top Movers Today")
gainers, losers = get_top_movers(WATCHLIST)

def color_table(x):
    if x.name == 'Change%':
        return ['background-color: lightgreen' if v > 0 else 'background-color: lightcoral' for v in x]
    return [''] * len(x)

if not gainers.empty:
    st.write("Top Gainers")
    st.dataframe(gainers.style.apply(color_table, axis=0))
    st.write("Top Losers")
    st.dataframe(losers.style.apply(color_table, axis=0))
else:
    st.info("No movers available yet.")

# Alerts for Big Moves
st.subheader("🚨 Alerts for Big Moves")
alerts = []
for ticker in WATCHLIST:
    hist = yf.Ticker(ticker).history(period="1d")
    if not hist.empty:
        change = (hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1] * 100
        if abs(change) >= BIG_MOVE_THRESHOLD:
            alerts.append(f"{ticker} moved {change:.2f}% today")
if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.info("No big moves currently.")

# News Feed
st.subheader(f"📰 Latest News for {selected_ticker}")
news_articles = get_news(selected_ticker, keywords=["budget", "RBI", "IPO"])
if news_articles:
    for title, source, published in news_articles:
        st.write(f"**{title}** ({source}, {published})")
else:
    st.info("No recent news found.")
