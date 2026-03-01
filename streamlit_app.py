import streamlit as st
import pandas as pd
import numpy as np

st.title("Stock Score Calculator")

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    prices = prices[-31:]

    logp = np.log(prices)
    returns = np.diff(logp)

    momentum = np.mean(returns)

    half = len(returns)//2
    curvature = np.mean(returns[-half:]) - np.mean(returns[:half])

    volatility = np.std(returns)

    score = (momentum + 0.5*curvature)/volatility

    st.write("Score:", score)
    st.write("Momentum:", momentum)
    st.write("Curvature:", curvature)
    st.write("Volatility:", volatility)
