import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Momentum Model",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; background-color: #060a12; color: #e2e8f0; }
  .stApp { background-color: #060a12; }
  .metric-card { background: #0a0e1a; border: 1px solid #0f1623; border-radius: 12px; padding: 16px 20px; margin-bottom: 8px; }
  .sig-strong-sell { color: #FF2D55; font-weight: 700; }
  .sig-sell        { color: #FF6B35; font-weight: 700; }
  .sig-watch-sell  { color: #FFB800; font-weight: 700; }
  .sig-strong-buy  { color: #00E5A0; font-weight: 700; }
  .sig-buy         { color: #00C4FF; font-weight: 700; }
  .sig-watch-buy   { color: #A78BFA; font-weight: 700; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def generate_signals(df):
    df = df.copy()
    df["rsi"]  = calc_rsi(df["Close"])
    df["macd"], df["macd_sig"], df["hist"] = calc_macd(df["Close"])
    df["prev_hist"] = df["hist"].shift(1)

    def classify(row):
        r, h, ph = row["rsi"], row["hist"], row["prev_hist"]
        bear_x = pd.notna(h) and pd.notna(ph) and ph > 0 and h <= 0
        bull_x = pd.notna(h) and pd.notna(ph) and ph < 0 and h >= 0
        if   bear_x and r > 70:  return "STRONG_SELL"
        elif bear_x:              return "SELL"
        elif pd.notna(r) and r > 75: return "WATCH_SELL"
        elif bull_x and r < 30:  return "STRONG_BUY"
        elif bull_x:              return "BUY"
        elif pd.notna(r) and r < 25: return "WATCH_BUY"
        return None

    df["signal"] = df.apply(classify, axis=1)
    return df

SIG = {
    "STRONG_SELL": {"label": "🔴 強売シグナル", "color": "#FF2D55"},
    "SELL":        {"label": "🟠 売シグナル",   "color": "#FF6B35"},
    "WATCH_SELL":  {"label": "🟡 売り注意",     "color": "#FFB800"},
    "STRONG_BUY":  {"label": "🟢 強買シグナル", "color": "#00E5A0"},
    "BUY":         {"label": "🔵 買シグナル",   "color": "#00C4FF"},
    "WATCH_BUY":   {"label": "🟣 買い注意",     "color": "#A78BFA"},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ GOLD MOMENTUM")
    st.markdown("---")
    ticker = st.text_input("ティッカー", value="GC=F", help="GC=F / GLD / IAU / XAUUSD=X").upper()
    period_map = {"3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}
    period_label = st.selectbox("取得期間", list(period_map.keys()), index=2)
    period = period_map[period_label]

    rsi_ob  = st.slider("RSI 買われ過ぎ", 60, 90, 70)
    rsi_os  = st.slider("RSI 売られ過ぎ", 10, 40, 30)
    st.markdown("---")
    fetch_btn = st.button("🔄 データ取得・更新", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**対応ティッカー例**")
    st.markdown("`GC=F` 金先物\n\n`GLD` SPDR金ETF\n\n`IAU` iShares金ETF\n\n`XAUUSD=X` 金スポット")
    st.markdown("---")
    st.caption("⚠ 教育・研究目的のみ\n実際の投資判断には使用しないでください")

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # 5分キャッシュ
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state or fetch_btn:
    with st.spinner(f"{ticker} のデータを取得中..."):
        raw = fetch_data(ticker, period)
    if raw is None or raw.empty:
        st.error(f"❌ {ticker} のデータ取得に失敗しました。ティッカーを確認してください。")
        st.stop()
    st.session_state.df     = raw
    st.session_state.ticker = ticker
    st.session_state.period = period_label
    st.session_state.fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df_raw = st.session_state.df
df = generate_signals(df_raw)

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"## ⬡ {st.session_state.ticker} — Gold Momentum Model")
    st.caption(f"RSI({rsi_ob}/{rsi_os}) + MACD(12/26/9)　|　{st.session_state.period}　|　{len(df)}営業日")
with col_status:
    st.success(f"✓ 取得済\n{st.session_state.get('fetched_at','')}")

st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
latest  = df.iloc[-1]
prev    = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
first   = df.iloc[0]
day_chg = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
prd_chg = (latest["Close"] - first["Close"]) / first["Close"] * 100
sig_rows = df[df["signal"].notna()].iloc[::-1]
last_sig = sig_rows.iloc[0] if not sig_rows.empty else None

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(f"{st.session_state.ticker} 現在値",
              f"${latest['Close']:.2f}",
              f"{day_chg:+.2f}% 前日比")
with k2:
    st.metric("期間リターン",
              f"{prd_chg:+.2f}%",
              f"${first['Close']:.2f} → ${latest['Close']:.2f}")
with k3:
    rsi_val = latest["rsi"]
    rsi_status = "⚠ 買われ過ぎ" if rsi_val > rsi_ob else "⚠ 売られ過ぎ" if rsi_val < rsi_os else "中立域"
    st.metric("RSI (14日)", f"{rsi_val:.1f}", rsi_status)
with k4:
    hist_val = latest["hist"]
    hist_dir = "上昇モメンタム ↑" if hist_val > 0 else "下降モメンタム ↓"
    st.metric("MACDヒストグラム", f"{hist_val:.3f}", hist_dir)
with k5:
    if last_sig is not None:
        sig_info = SIG[last_sig["signal"]]
        st.metric("最新シグナル", sig_info["label"], str(last_sig.name.date()))
    else:
        st.metric("最新シグナル", "なし", "—")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.22, 0.23],
    vertical_spacing=0.03,
    subplot_titles=["価格チャート + シグナル", "RSI (14)", "MACD (12/26/9)"]
)

CHART_BG   = "#060a12"
GRID_COLOR = "#0d1220"
TEXT_COLOR = "#475569"

# Price line
fig.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    mode="lines", name="終値",
    line=dict(color="#FFD700", width=2.5)
), row=1, col=1)

# Signal markers on price
for sig_key, info in SIG.items():
    sub = df[df["signal"] == sig_key]
    if sub.empty: continue
    is_sell = "SELL" in sig_key
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["Close"],
        mode="markers",
        name=info["label"],
        marker=dict(
            color=info["color"],
            size=12,
            symbol="triangle-up" if not is_sell else "triangle-down",
            line=dict(color=CHART_BG, width=1)
        ),
        hovertemplate=f"<b>{info['label']}</b><br>日付: %{{x|%Y-%m-%d}}<br>価格: $%{{y:.2f}}<extra></extra>"
    ), row=1, col=1)

# High/Low reference lines
max_p = df["Close"].max()
min_p = df["Close"].min()
fig.add_hline(y=max_p, line_dash="dash", line_color="#FF2D5544", row=1, col=1,
              annotation=dict(text=f"高値 ${max_p:.2f}", font=dict(color="#FF2D55", size=10)))
fig.add_hline(y=min_p, line_dash="dash", line_color="#00E5A044", row=1, col=1,
              annotation=dict(text=f"安値 ${min_p:.2f}", font=dict(color="#00E5A0", size=10)))

# RSI
fig.add_trace(go.Scatter(
    x=df.index, y=df["rsi"],
    mode="lines", name="RSI",
    line=dict(color="#A78BFA", width=1.8)
), row=2, col=1)
fig.add_hrect(y0=rsi_ob, y1=100, fillcolor="#FF2D55", opacity=0.06, row=2, col=1)
fig.add_hrect(y0=0, y1=rsi_os, fillcolor="#00E5A0", opacity=0.06, row=2, col=1)
fig.add_hline(y=rsi_ob, line_dash="dash", line_color="#FF2D5566", line_width=1, row=2, col=1)
fig.add_hline(y=rsi_os, line_dash="dash", line_color="#00E5A066", line_width=1, row=2, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="#1e2535", line_width=1, row=2, col=1)

# MACD histogram
colors = ["#00E5A0" if v >= 0 else "#FF2D55" for v in df["hist"].fillna(0)]
fig.add_trace(go.Bar(
    x=df.index, y=df["hist"],
    name="ヒストグラム",
    marker_color=colors, opacity=0.65
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["macd"],
    mode="lines", name="MACD",
    line=dict(color="#00C4FF", width=1.5)
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["macd_sig"],
    mode="lines", name="Signal",
    line=dict(color="#FFD700", width=1, dash="dash")
), row=3, col=1)
fig.add_hline(y=0, line_color="#1e2535", line_width=1, row=3, col=1)

# Layout
fig.update_layout(
    height=700,
    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    font=dict(family="IBM Plex Mono", color=TEXT_COLOR, size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
fig.update_yaxes(range=[0, 100], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Signal log ────────────────────────────────────────────────────────────────
st.markdown("### 📋 シグナル履歴")
if sig_rows.empty:
    st.info("シグナルなし（データが短い場合は35日以上必要）")
else:
    filter_cols = st.columns(len(SIG))
    filters = {}
    for i, (k, v) in enumerate(SIG.items()):
        with filter_cols[i]:
            filters[k] = st.checkbox(v["label"], value=True, key=f"f_{k}")

    active_sigs = [k for k, v in filters.items() if v]
    filtered = sig_rows[sig_rows["signal"].isin(active_sigs)]

    if filtered.empty:
        st.info("該当シグナルなし")
    else:
        rows = []
        for idx, row in filtered.iterrows():
            info = SIG[row["signal"]]
            rows.append({
                "日付": str(idx.date()),
                "シグナル": info["label"],
                "価格": f"${row['Close']:.2f}",
                "RSI": f"{row['rsi']:.1f}" if pd.notna(row['rsi']) else "—",
                "MACDヒスト": f"{row['hist']:.3f}" if pd.notna(row['hist']) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
