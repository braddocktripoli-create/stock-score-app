import streamlit as st
import pandas as pd
import numpy as np
import re

st.title("Stock Score Calculator (Paste Friendly)")

st.write("Paste rows like: `Sep 30, 2025  222.03  222.24  217.89  219.57  219.57  48,396,400`")

text = st.text_area("Paste data here", height=260, placeholder="Paste your rows here...")

lookback = st.number_input("Lookback days", min_value=10, max_value=252, value=30, step=1)
alpha = st.number_input("Alpha (curvature weight)", min_value=0.0, max_value=2.0, value=0.5, step=0.05)

def parse_pasted_rows(raw: str) -> pd.DataFrame:
    """
    Parses pasted Yahoo-style rows:
    Sep 30, 2025  222.03  222.24  217.89  219.57  219.57  48,396,400
    Works with tabs or multiple spaces. Volume commas are fine.
    """
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Split by tabs OR 2+ spaces (keeps the date chunk intact)
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 7:
            # If user pasted with single spaces, try a fallback: split by whitespace
            parts = line.split()
            # Now date is 3 tokens like: Sep 30, 2025
            # Rebuild if needed
            if len(parts) >= 9:
                date_str = " ".join(parts[:3])
                nums = parts[3:]
                parts = [date_str] + nums
            else:
                raise ValueError(f"Couldn't parse this line (need 7 columns): {line}")

        date_str = parts[0]
        open_, high, low, close, adj_close, volume = parts[1:7]

        # Clean volume commas
        volume = volume.replace(",", "")

        rows.append({
            "Date": date_str,
            "Open": float(open_),
            "High": float(high),
            "Low": float(low),
            "Close": float(close),
            "Adj Close": float(adj_close),
            "Volume": int(float(volume))
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

def compute_score(df: pd.DataFrame, lookback_days: int, alpha: float):
    if "Adj Close" in df.columns:
        prices = pd.to_numeric(df["Adj Close"], errors="coerce").dropna().to_numpy()
        price_col = "Adj Close"
    else:
        prices = pd.to_numeric(df["Close"], errors="coerce").dropna().to_numpy()
        price_col = "Close"

    if len(prices) < lookback_days + 2:
        raise ValueError(f"Not enough rows. Need at least {lookback_days+2} rows, got {len(prices)}.")

    prices = prices[-(lookback_days + 1):]
    logp = np.log(prices)
    r = np.diff(logp)  # log returns

    M = float(np.mean(r))  # momentum
    half = max(1, len(r) // 2)
    C = float(np.mean(r[-half:]) - np.mean(r[:half]))  # curvature proxy
    sigma = float(np.std(r, ddof=1)) if len(r) > 1 else float(np.std(r))  # volatility

    if sigma == 0:
        Si = float("inf") if (M + alpha * C) > 0 else float("-inf")
    else:
        Si = float((M + alpha * C) / sigma)

    return price_col, M, C, sigma, Si

if text.strip():
    try:
        df = parse_pasted_rows(text)
        st.success(f"Parsed {len(df)} rows.")
        st.dataframe(df.tail(10), use_container_width=True)

        price_col, M, C, sigma, Si = compute_score(df, int(lookback), float(alpha))

        st.subheader("Results")
        st.write(f"**Price column used:** {price_col}")
        st.metric("Score (Si)", f"{Si:.6f}")
        st.write({
            "Momentum (mean log return)": M,
            "Curvature (change in mean return)": C,
            "Volatility (std of log returns)": sigma
        })

        st.line_chart(df.set_index("Date")[price_col])

    except Exception as e:
        st.error(str(e))
