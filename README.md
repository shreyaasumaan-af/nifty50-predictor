# Nifty 50 Portfolio Analyzer

Predict tomorrow's stock movement for your personal Nifty 50 holdings using Machine Learning.

## Setup (one time only)

1. Make sure Python is installed → python.org
2. Open terminal / command prompt in this folder
3. Install libraries:

```
pip install -r requirements.txt
```

## Run the app

```
streamlit run app.py
```

Your browser will open automatically at http://localhost:8501

## How it works

1. Enter your Nifty 50 stocks, quantity, and buy price
2. Click "Analyze my portfolio"
3. The app fetches 2 years of historical data for each stock
4. Calculates RSI, MACD, Bollinger Bands, SMA as features
5. Trains a Random Forest model on that data
6. Predicts whether the stock will go UP or DOWN tomorrow
7. Shows your P&L alongside the prediction

## Deploy online (free)

1. Create account at share.streamlit.io
2. Push this folder to GitHub
3. Connect your GitHub repo on Streamlit Cloud
4. Click Deploy — done!

## Disclaimer

This project is for educational purposes only. Not financial advice.
