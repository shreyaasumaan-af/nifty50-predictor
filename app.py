import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Nifty 50 Portfolio Analyzer")
st.caption("Enter your holdings below — we'll fetch live prices and predict tomorrow's movement using ML.")

# ── Nifty 50 stock list ───────────────────────────────────────────────────────
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NESTLEIND.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TECHM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "M&M.NS",
    "BAJAJ-AUTO.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "SHRIRAMFIN.NS", "TATACONSUM.NS", "UPL.NS", "VEDL.NS"
]

# ── Helper: fetch data & engineer features ────────────────────────────────────
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 60:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    bb = BollingerBands(close=close, window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_pct"] = bb.bollinger_pband()
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["Price_change"] = close.pct_change()
    df["Volume_ratio"] = volume / volume.rolling(20).mean()
    df["High_Low_pct"] = (df["High"].squeeze() - df["Low"].squeeze()) / close

    df["Target"] = (close.shift(-1) > close).astype(int)
    df.dropna(inplace=True)
    return df


def train_and_predict(df):
    features = [
        "RSI", "MACD", "MACD_signal", "MACD_diff",
        "BB_pct", "SMA_20", "SMA_50",
        "Price_change", "Volume_ratio", "High_Low_pct"
    ]
    X = df[features].iloc[:-1]
    y = df["Target"].iloc[:-1]
    X_latest = df[features].iloc[[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    pred = model.predict(X_latest)[0]
    proba = model.predict_proba(X_latest)[0][pred]
    return pred, round(proba * 100, 1), round(acc * 100, 1)


@st.cache_data(ttl=600)
def get_current_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="2d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except:
        pass
    return None

# ── Portfolio input ───────────────────────────────────────────────────────────
st.subheader("Your holdings")

if "holdings" not in st.session_state:
    st.session_state.holdings = [
        {"stock": "RELIANCE.NS", "qty": 10, "buy_price": 2850.0},
        {"stock": "TCS.NS",      "qty": 5,  "buy_price": 3620.0},
    ]

def add_row():
    st.session_state.holdings.append({"stock": NIFTY50_STOCKS[0], "qty": 1, "buy_price": 0.0})

def remove_row(i):
    st.session_state.holdings.pop(i)

for i, h in enumerate(st.session_state.holdings):
    cols = st.columns([3, 1.5, 2, 0.6])
    with cols[0]:
        h["stock"] = st.selectbox("Stock", NIFTY50_STOCKS,
                                  index=NIFTY50_STOCKS.index(h["stock"]) if h["stock"] in NIFTY50_STOCKS else 0,
                                  key=f"stock_{i}", label_visibility="collapsed")
    with cols[1]:
        h["qty"] = st.number_input("Qty", min_value=1, value=h["qty"],
                                   key=f"qty_{i}", label_visibility="collapsed")
    with cols[2]:
        h["buy_price"] = st.number_input("Buy price ₹", min_value=0.0, value=float(h["buy_price"]),
                                         key=f"bp_{i}", label_visibility="collapsed")
    with cols[3]:
        if st.button("✕", key=f"del_{i}"):
            remove_row(i)
            st.rerun()

st.button("+ Add stock", on_click=add_row)

# ── Analyze button ────────────────────────────────────────────────────────────
if st.button("Analyze my portfolio", type="primary", use_container_width=True):
    if not st.session_state.holdings:
        st.warning("Please add at least one stock.")
        st.stop()

    st.divider()
    st.subheader("Portfolio summary")

    results = []
    total_invested = 0.0
    total_current = 0.0

    progress = st.progress(0, text="Fetching data and running predictions...")

    for idx, h in enumerate(st.session_state.holdings):
        ticker = h["stock"]
        qty    = h["qty"]
        bp     = h["buy_price"]

        progress.progress((idx + 1) / len(st.session_state.holdings),
                          text=f"Analyzing {ticker}...")

        current_price = get_current_price(ticker)
        if current_price is None:
            st.warning(f"Could not fetch price for {ticker}, skipping.")
            continue

        invested = bp * qty
        current  = current_price * qty
        pnl_amt  = current - invested
        pnl_pct  = ((current_price - bp) / bp * 100) if bp > 0 else 0

        df = get_stock_data(ticker)
        if df is not None:
            pred, confidence, model_acc = train_and_predict(df)
        else:
            pred, confidence, model_acc = None, None, None

        total_invested += invested
        total_current  += current

        results.append({
            "ticker":       ticker,
            "qty":          qty,
            "buy_price":    bp,
            "current":      current_price,
            "invested":     invested,
            "current_val":  current,
            "pnl_amt":      pnl_amt,
            "pnl_pct":      pnl_pct,
            "pred":         pred,
            "confidence":   confidence,
            "model_acc":    model_acc,
        })

    progress.empty()

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_pnl     = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total invested",   f"₹{total_invested:,.0f}")
    c2.metric("Current value",    f"₹{total_current:,.0f}")
    c3.metric("Total P&L",        f"₹{total_pnl:,.0f}",
              delta=f"{total_pnl_pct:.1f}%")
    c4.metric("Stocks held",      len(results))

    # ── Per-stock cards ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Stock-wise predictions")

    for r in results:
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                st.markdown(f"**{r['ticker']}**")
                st.caption(f"{r['qty']} shares  ·  bought @ ₹{r['buy_price']:,.2f}")
                st.write(f"Current price: **₹{r['current']:,.2f}**")

            with col2:
                color = "normal" if r["pnl_amt"] >= 0 else "inverse"
                st.metric("P&L",
                          f"₹{r['pnl_amt']:,.0f}",
                          delta=f"{r['pnl_pct']:.1f}%",
                          delta_color=color)

            with col3:
                if r["pred"] is not None:
                    direction = "⬆ UP" if r["pred"] == 1 else "⬇ DOWN"
                    badge     = "🟢" if r["pred"] == 1 else "🔴"
                    st.markdown(f"**Tomorrow's prediction**")
                    st.markdown(f"### {badge} {direction}")
                    st.caption(f"Confidence: {r['confidence']}%  ·  Model accuracy: {r['model_acc']}%")
                else:
                    st.info("Not enough data to predict.")

    # ── Portfolio breakdown table ─────────────────────────────────────────────
    st.divider()
    st.subheader("Full breakdown")

    table_data = []
    for r in results:
        table_data.append({
            "Stock":        r["ticker"],
            "Qty":          r["qty"],
            "Buy price":    f"₹{r['buy_price']:,.2f}",
            "Current":      f"₹{r['current']:,.2f}",
            "Invested":     f"₹{r['invested']:,.0f}",
            "Value now":    f"₹{r['current_val']:,.0f}",
            "P&L":          f"₹{r['pnl_amt']:,.0f}",
            "P&L %":        f"{r['pnl_pct']:.1f}%",
            "Prediction":   ("UP" if r["pred"] == 1 else "DOWN") if r["pred"] is not None else "N/A",
            "Confidence":   f"{r['confidence']}%" if r["confidence"] else "N/A",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.caption("⚠️ Predictions are for educational purposes only. Not financial advice.")
