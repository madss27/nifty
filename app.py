from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator


SYMBOL = "^NSEI"
MARKET_NAME = "NIFTY 50"
EMA_WINDOW = 33
RSI_WINDOW = 14


st.set_page_config(
    page_title="NIFTY Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css(theme: str) -> None:
    if theme == "Light":
        bg = "#eef3f5"
        panel = "#ffffff"
        panel_2 = "#f5f8fa"
        text = "#17212b"
        muted = "#5d6a75"
        line = "rgba(23, 33, 43, 0.12)"
    else:
        bg = "#101418"
        panel = "#171d23"
        panel_2 = "#1d252d"
        text = "#e7edf2"
        muted = "#9aa8b4"
        line = "rgba(231, 237, 242, 0.11)"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {bg};
                color: {text};
            }}
            [data-testid="stSidebar"] {{
                background: {panel};
                border-right: 1px solid {line};
            }}
            [data-testid="stHeader"] {{
                background: transparent;
            }}
            .block-container {{
                max-width: 1420px;
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }}
            .top-row {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .title {{
                font-size: clamp(2rem, 5vw, 4rem);
                font-weight: 850;
                line-height: 1;
                color: {text};
            }}
            .subtitle {{
                color: {muted};
                font-size: 1.05rem;
                margin-top: 0.45rem;
            }}
            .pill {{
                border: 1px solid {line};
                background: {panel};
                color: {muted};
                border-radius: 999px;
                padding: 0.6rem 0.9rem;
                white-space: nowrap;
            }}
            .card {{
                background: {panel};
                border: 1px solid {line};
                border-radius: 8px;
                padding: 1.05rem;
                min-height: 118px;
                box-shadow: 0 14px 34px rgba(0,0,0,0.12);
            }}
            .metric-label {{
                color: {muted};
                font-size: 0.92rem;
                margin-bottom: 0.35rem;
            }}
            .metric-value {{
                color: {text};
                font-size: clamp(1.7rem, 3vw, 2.45rem);
                font-weight: 800;
                line-height: 1.08;
            }}
            .metric-note {{
                color: {muted};
                font-size: 0.86rem;
                margin-top: 0.45rem;
            }}
            .signal-card {{
                background: {panel};
                border: 1px solid {line};
                border-left: 7px solid var(--signal);
                border-radius: 8px;
                padding: 1.2rem;
                min-height: 210px;
            }}
            .signal-row {{
                display: flex;
                align-items: center;
                gap: 0.9rem;
            }}
            .signal-light {{
                width: 21px;
                height: 21px;
                border-radius: 999px;
                background: var(--signal);
                box-shadow: 0 0 0 7px color-mix(in srgb, var(--signal) 18%, transparent),
                            0 0 28px color-mix(in srgb, var(--signal) 35%, transparent);
            }}
            .signal-title {{
                color: var(--signal);
                font-size: clamp(2rem, 4vw, 3.4rem);
                font-weight: 900;
                line-height: 1;
            }}
            .suggestion {{
                margin-top: 1rem;
                color: {text};
                font-size: 1.15rem;
            }}
            .explain {{
                margin-top: 0.5rem;
                color: {muted};
                font-size: 0.98rem;
            }}
            .section-label {{
                color: {muted};
                text-transform: uppercase;
                font-size: 0.78rem;
                letter-spacing: 0.08rem;
                margin: 0.4rem 0 0.6rem 0;
            }}
            .trade-box {{
                background: {panel_2};
                border: 1px solid {line};
                border-radius: 8px;
                padding: 1rem;
                min-height: 120px;
            }}
            .trade-box strong {{
                display: block;
                color: {text};
                font-size: 1.2rem;
                margin-bottom: 0.35rem;
            }}
            .trade-box span {{
                color: {muted};
            }}
            @media (max-width: 760px) {{
                .top-row {{
                    display: block;
                }}
                .pill {{
                    display: inline-block;
                    margin-top: 0.8rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5, show_spinner=False)
def fetch_market_data() -> pd.DataFrame:
    """Fetch near-real-time NIFTY 50 candles from Yahoo Finance."""
    ticker = yf.Ticker(SYMBOL)
    data = ticker.history(
        period="5d",
        interval="1m",
        auto_adjust=False,
        prepost=False,
        actions=False,
    )

    if data is None or data.empty:
        raise RuntimeError("No market data returned. The market may be closed or Yahoo Finance may be unavailable.")

    data = data.reset_index()
    data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]

    if "datetime" not in data.columns and "date" in data.columns:
        data = data.rename(columns={"date": "datetime"})

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing columns from data feed: {missing}")

    if "volume" not in data.columns:
        data["volume"] = 0

    data = data.dropna(subset=["open", "high", "low", "close"]).copy()
    data["datetime"] = pd.to_datetime(data["datetime"])

    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime")


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA(33), RSI(14), and VWAP/reference VWAP."""
    df = data.copy()

    df["ema_33"] = EMAIndicator(close=df["close"], window=EMA_WINDOW).ema_indicator()
    df["ema_33"] = df["ema_33"].ffill().fillna(df["close"])

    df["rsi"] = RSIIndicator(close=df["close"], window=RSI_WINDOW).rsi()
    df["rsi"] = df["rsi"].fillna(50)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].fillna(0)

    # Index feeds often have missing/zero volume. In that case, use a stable
    # typical-price VWAP proxy so the dashboard still gives a useful reference.
    if volume.gt(0).any():
        cumulative_volume = volume.cumsum().replace(0, np.nan)
        df["vwap"] = ((typical_price * volume).cumsum() / cumulative_volume).ffill()
        df["vwap_source"] = "volume"
    else:
        df["vwap"] = typical_price.expanding().mean()
        df["vwap_source"] = "typical-price proxy"

    return df


def evaluate_signal(row: pd.Series) -> dict:
    price = float(row["close"])
    ema = float(row["ema_33"])
    rsi = float(row["rsi"])
    vwap = float(row["vwap"])

    if price > ema and rsi > 45 and price > vwap:
        return {
            "bias": "Bullish",
            "headline": "BULLISH SIGNAL",
            "color": "#69b893",
            "suggestion": "Buy suggestion: consider long setups after your own risk checks.",
            "explanation": "Price is above EMA(33), RSI is above 45, and price is above VWAP.",
            "action": "BUY",
        }

    if price < ema and rsi < 45 and price < vwap:
        return {
            "bias": "Bearish",
            "headline": "BEARISH SIGNAL",
            "color": "#d96d72",
            "suggestion": "Sell suggestion: consider short or exit setups after your own risk checks.",
            "explanation": "Price is below EMA(33), RSI is below 45, and price is below VWAP.",
            "action": "SELL",
        }

    return {
        "bias": "Neutral",
        "headline": "NEUTRAL MARKET",
        "color": "#9aa8b4",
        "suggestion": "No clear trade suggestion. Wait for all conditions to align.",
        "explanation": "The bullish or bearish rules are not fully aligned yet.",
        "action": "WAIT",
    }


def format_number(value: float) -> str:
    return f"{value:,.2f}"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal(signal: dict) -> None:
    st.markdown(
        f"""
        <div class="signal-card" style="--signal: {signal['color']};">
            <div class="signal-row">
                <div class="signal-light"></div>
                <div class="signal-title">{signal['headline']}</div>
            </div>
            <div class="suggestion">{signal['suggestion']}</div>
            <div class="explain">{signal['explanation']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_buy_sell_boxes(active_bias: str) -> None:
    buy_border = "#69b893" if active_bias == "Bullish" else "rgba(150,150,150,0.25)"
    sell_border = "#d96d72" if active_bias == "Bearish" else "rgba(150,150,150,0.25)"

    st.markdown('<div class="section-label">Action panels</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="trade-box" style="border-color:{buy_border};">
                <strong>BUY Zone</strong>
                <span>Active when price is above EMA(33), RSI is above 45, and price is above VWAP.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="trade-box" style="border-color:{sell_border};">
                <strong>SELL Zone</strong>
                <span>Active when price is below EMA(33), RSI is below 45, and price is below VWAP.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chart(data: pd.DataFrame, theme: str) -> None:
    if theme == "Light":
        panel = "#ffffff"
        text = "#17212b"
        grid = "rgba(23,33,43,0.12)"
    else:
        panel = "#171d23"
        text = "#e7edf2"
        grid = "rgba(231,237,242,0.10)"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
        subplot_titles=("Candles with EMA(33) and VWAP", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=data["datetime"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=MARKET_NAME,
            increasing_line_color="#69b893",
            decreasing_line_color="#d96d72",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data["datetime"],
            y=data["ema_33"],
            name="EMA 33",
            line=dict(color="#5f7ea5", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data["datetime"],
            y=data["vwap"],
            name="VWAP",
            line=dict(color="#d4ad63", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data["datetime"],
            y=data["rsi"],
            name="RSI",
            line=dict(color="#69b893", width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=45, line_dash="dot", line_color="#9aa8b4", row=2, col=1)

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor=panel,
        plot_bgcolor=panel,
        font=dict(color=text, size=13),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(gridcolor=grid, showline=False)
    fig.update_yaxes(gridcolor=grid, showline=False)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def add_history(row: pd.Series, signal: dict) -> None:
    if "signal_history" not in st.session_state:
        st.session_state.signal_history = []

    market_time = pd.to_datetime(row["datetime"]).strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_time": market_time,
        "price": round(float(row["close"]), 2),
        "ema_33": round(float(row["ema_33"]), 2),
        "rsi": round(float(row["rsi"]), 2),
        "vwap": round(float(row["vwap"]), 2),
        "bias": signal["bias"],
        "suggestion": signal["action"],
    }

    if st.session_state.signal_history:
        last = st.session_state.signal_history[-1]
        if last["market_time"] == entry["market_time"] and last["bias"] == entry["bias"]:
            return

    st.session_state.signal_history.append(entry)
    st.session_state.signal_history = st.session_state.signal_history[-100:]


with st.sidebar:
    st.markdown("## Controls")
    theme = st.segmented_control("Theme", ["Dark", "Light"], default="Dark")
    refresh_seconds = st.slider("Auto-refresh seconds", 5, 60, 5, step=5)
    show_raw = st.toggle("Show latest raw candles", value=False)

    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("## Signal Rules")
    st.caption("Bullish: Price > EMA(33), RSI > 45, Price > VWAP")
    st.caption("Bearish: Price < EMA(33), RSI < 45, Price < VWAP")

    st.markdown("## Data")
    st.caption("Source: Yahoo Finance via yfinance. Index data may be delayed.")


inject_css(theme)
st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")

try:
    raw_data = fetch_market_data()
    data = add_indicators(raw_data)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

latest = data.iloc[-1]
signal = evaluate_signal(latest)
add_history(latest, signal)

now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M:%S %p IST")

st.markdown(
    f"""
    <div class="top-row">
        <div>
            <div class="title">{MARKET_NAME} Live Dashboard</div>
            <div class="subtitle">Simple live trend, momentum, and VWAP view for cleaner trading decisions.</div>
        </div>
        <div class="pill">Last update: {now_ist} | Feed: Yahoo Finance {SYMBOL}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This dashboard is for market monitoring and education, not financial advice. Confirm trades with your own risk plan.",
    icon=None,
)

cols = st.columns(4)
with cols[0]:
    metric_card("Current NIFTY", format_number(float(latest["close"])), "Latest fetched candle")
with cols[1]:
    metric_card("EMA(33)", format_number(float(latest["ema_33"])), "Trend reference")
with cols[2]:
    metric_card("RSI(14)", format_number(float(latest["rsi"])), "Threshold: 45")
with cols[3]:
    metric_card("VWAP", format_number(float(latest["vwap"])), f"Source: {latest['vwap_source']}")

st.write("")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    render_signal(signal)
with right:
    render_buy_sell_boxes(signal["bias"])

st.write("")
render_chart(data.tail(240), theme)

st.markdown('<div class="section-label">Signal history</div>', unsafe_allow_html=True)
history = pd.DataFrame(st.session_state.signal_history)

if history.empty:
    st.caption("Signal history will appear after the first refresh.")
else:
    st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)

if show_raw:
    st.markdown('<div class="section-label">Latest raw candles</div>', unsafe_allow_html=True)
    st.dataframe(data.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
