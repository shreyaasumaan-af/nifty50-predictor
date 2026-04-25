import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Nifty 50 Portfolio Analyzer", page_icon="📈", layout="wide")
st.title("📈 Nifty 50 Portfolio Analyzer")
st.caption("Enter your holdings — we'll fetch live prices and predict tomorrow's movement.")

NIFTY50 = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","WIPRO.NS","HCLTECH.NS","NTPC.NS","ONGC.NS","TATAMOTORS.NS","TATASTEEL.NS","BAJFINANCE.NS","ADANIPORTS.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","BAJAJ-AUTO.NS","BRITANNIA.NS","CIPLA.NS","TATACONSUM.NS"]

@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty or len(df) < 30:
        return None
    close = df["Close"].squeeze()
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    df["MACD"] = MACD(close=close).macd_diff()
    df["SMA20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["Change"] = close.pct_change()
    df.dropna(inplace=True)
    return df

def predict(ticker):
    df = get_data(ticker)
    if df is None:
        return None, None
    latest = df.iloc[-1]
    score = 0
    if float(latest["RSI"]) < 50:
    score += 1: score += 1
    if latest["MACD"] > 0: score += 1
    if latest["SMA20"] > latest["SMA50"]: score += 1
    if latest["Change"] > 0: score += 1
    confidence = int((score / 4) * 100)
    pred = "UP" if score >= 2 else "DOWN"
    return pred, confidence

@st.cache_data(ttl=600)
def get_price(ticker):
    try:
        return round(float(yf.Ticker(ticker).history(period="2d")["Close"].iloc[-1]), 2)
    except:
        return None

if "holdings" not in st.session_state:
    st.session_state.holdings = [{"stock": "RELIANCE.NS", "qty": 10, "buy_price": 2850.0}]

st.subheader("Your holdings")
for i, h in enumerate(st.session_state.holdings):
    cols = st.columns([3, 1.5, 2, 0.6])
    h["stock"] = cols[0].selectbox("Stock", NIFTY50, index=NIFTY50.index(h["stock"]) if h["stock"] in NIFTY50 else 0, key=f"s{i}", label_visibility="collapsed")
    h["qty"] = cols[1].number_input("Qty", min_value=1, value=h["qty"], key=f"q{i}", label_visibility="collapsed")
    h["buy_price"] = cols[2].number_input("Buy price", min_value=0.0, value=float(h["buy_price"]), key=f"b{i}", label_visibility="collapsed")
    if cols[3].button("✕", key=f"d{i}"):
        st.session_state.holdings.pop(i)
        st.rerun()

if st.button("+ Add stock"):
    st.session_state.holdings.append({"stock": "TCS.NS", "qty": 1, "buy_price": 0.0})
    st.rerun()

if st.button("Analyze my portfolio", type="primary", use_container_width=True):
    total_invested, total_current = 0, 0
    results = []
    for h in st.session_state.holdings:
        price = get_price(h["stock"])
        if not price: continue
        invested = h["buy_price"] * h["qty"]
        current = price * h["qty"]
        pnl = current - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0
        pred, conf = predict(h["stock"])
        total_invested += invested
        total_current += current
        results.append({"ticker": h["stock"], "price": price, "invested": invested, "current": current, "pnl": pnl, "pnl_pct": pnl_pct, "pred": pred, "conf": conf})

    st.divider()
    st.subheader("Portfolio summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total invested", f"₹{total_invested:,.0f}")
    c2.metric("Current value", f"₹{total_current:,.0f}")
    c3.metric("Total P&L", f"₹{total_current - total_invested:,.0f}", delta=f"{((total_current-total_invested)/total_invested*100):.1f}%" if total_invested > 0 else "0%")

    st.divider()
    st.subheader("Predictions")
    for r in results:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{r['ticker']}**")
            c1.write(f"₹{r['price']:,.2f}")
            c2.metric("P&L", f"₹{r['pnl']:,.0f}", delta=f"{r['pnl_pct']:.1f}%")
            if r["pred"]:
                c3.markdown("**Tomorrow's prediction**")
                c3.markdown(f"### {'🟢 UP' if r['pred'] == 'UP' else '🔴 DOWN'}")
                c3.caption(f"Confidence: {r['conf']}%")

    st.caption("⚠️ For educational purposes only. Not financial advice.")
