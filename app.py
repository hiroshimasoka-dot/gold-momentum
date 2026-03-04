import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Momentum Model",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; background-color: #060a12; color: #e2e8f0; }
  .stApp { background-color: #060a12; }
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
        if   bear_x and r > 70:       return "STRONG_SELL"
        elif bear_x:                   return "SELL"
        elif pd.notna(r) and r > 75:  return "WATCH_SELL"
        elif bull_x and r < 30:       return "STRONG_BUY"
        elif bull_x:                   return "BUY"
        elif pd.notna(r) and r < 25:  return "WATCH_BUY"
        return None
    df["signal"] = df.apply(classify, axis=1)
    return df

SIG = {
    "STRONG_SELL": {"label": "🔴 強売シグナル", "color": "#FF2D55", "is_sell": True},
    "SELL":        {"label": "🟠 売シグナル",   "color": "#FF6B35", "is_sell": True},
    "WATCH_SELL":  {"label": "🟡 売り注意",     "color": "#FFB800", "is_sell": True},
    "STRONG_BUY":  {"label": "🟢 強買シグナル", "color": "#00E5A0", "is_sell": False},
    "BUY":         {"label": "🔵 買シグナル",   "color": "#00C4FF", "is_sell": False},
    "WATCH_BUY":   {"label": "🟣 買い注意",     "color": "#A78BFA", "is_sell": False},
}

# ── Email sender ──────────────────────────────────────────────────────────────
def send_email(gmail_user, gmail_app_password, to_email, subject, body):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_user
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_app_password)
            server.sendmail(gmail_user, to_email, msg.as_string())
        return True, "送信成功"
    except smtplib.SMTPAuthenticationError:
        return False, "認証エラー：Gmailアドレスとアプリパスワードを確認してください"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def build_email_body(ticker, signal_key, price, rsi, hist, date_str):
    info  = SIG[signal_key]
    color = info["color"]
    label = info["label"]
    direction = "売却を検討" if info["is_sell"] else "購入を検討"
    return f"""
<div style="font-family:monospace;background:#060a12;color:#e2e8f0;padding:32px;border-radius:12px;max-width:480px">
  <div style="font-size:22px;font-weight:800;color:#FFD700;letter-spacing:0.1em;margin-bottom:4px">⬡ GOLD MOMENTUM ALERT</div>
  <div style="font-size:11px;color:#475569;margin-bottom:24px;letter-spacing:0.15em">RSI + MACD モメンタムモデル</div>
  <div style="background:#0a0e1a;border-left:4px solid {color};border-radius:8px;padding:20px;margin-bottom:20px">
    <div style="font-size:28px;font-weight:800;color:{color};margin-bottom:8px">{label}</div>
    <div style="font-size:32px;font-weight:800;color:#FFD700">${price:.2f}</div>
    <div style="font-size:12px;color:#64748b;margin-top:4px">{ticker} · {date_str}</div>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr style="border-bottom:1px solid #1e2535">
      <td style="padding:10px 0;color:#64748b">RSI (14日)</td>
      <td style="padding:10px 0;text-align:right;font-weight:700;color:#A78BFA">{rsi:.1f}</td>
    </tr>
    <tr style="border-bottom:1px solid #1e2535">
      <td style="padding:10px 0;color:#64748b">MACDヒストグラム</td>
      <td style="padding:10px 0;text-align:right;font-weight:700;color:#00C4FF">{hist:.3f}</td>
    </tr>
    <tr>
      <td style="padding:10px 0;color:#64748b">推奨アクション</td>
      <td style="padding:10px 0;text-align:right;font-weight:700;color:{color}">{direction}</td>
    </tr>
  </table>
  <div style="margin-top:24px;font-size:10px;color:#1e2535">⚠ このアラートは教育・研究目的のみです。実際の投資判断には使用しないでください。</div>
</div>
"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ GOLD MOMENTUM")
    st.markdown("---")
    ticker = st.text_input("ティッカー", value="GC=F").upper()
    period_map = {"3ヶ月": "3mo", "6ヶ月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}
    period_label = st.selectbox("取得期間", list(period_map.keys()), index=2)
    period = period_map[period_label]
    rsi_ob = st.slider("RSI 買われ過ぎ", 60, 90, 70)
    rsi_os = st.slider("RSI 売られ過ぎ", 10, 40, 30)
    st.markdown("---")
    fetch_btn = st.button("🔄 データ取得・更新", use_container_width=True, type="primary")

    # ── Email settings ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📧 メール通知")
    with st.expander("Gmail設定", expanded=False):
        st.markdown("""
**Gmailアプリパスワードの取得方法：**
1. [myaccount.google.com](https://myaccount.google.com) を開く
2. セキュリティ → 2段階認証をオン
3. 「アプリパスワード」を検索
4. アプリ:「メール」→ 生成される16桁をコピー
        """)
    gmail_user = st.text_input("Gmailアドレス", placeholder="your@gmail.com")
    gmail_pass = st.text_input("アプリパスワード (16桁)", type="password", placeholder="xxxx xxxx xxxx xxxx")
    notify_to  = st.text_input("送信先メール", placeholder="送り先のメールアドレス")

    notify_strong_sell = st.checkbox("🔴 強売シグナル", value=True)
    notify_sell        = st.checkbox("🟠 売シグナル",   value=True)
    notify_watch_sell  = st.checkbox("🟡 売り注意",     value=False)
    notify_strong_buy  = st.checkbox("🟢 強買シグナル", value=True)
    notify_buy         = st.checkbox("🔵 買シグナル",   value=True)
    notify_watch_buy   = st.checkbox("🟣 買い注意",     value=False)

    notify_map = {
        "STRONG_SELL": notify_strong_sell,
        "SELL":        notify_sell,
        "WATCH_SELL":  notify_watch_sell,
        "STRONG_BUY":  notify_strong_buy,
        "BUY":         notify_buy,
        "WATCH_BUY":   notify_watch_buy,
    }

    test_btn = st.button("📨 テスト送信", use_container_width=True)
    st.markdown("---")
    st.caption("⚠ 教育・研究目的のみ")

# ── Test email ────────────────────────────────────────────────────────────────
if test_btn:
    if not gmail_user or not gmail_pass or not notify_to:
        st.sidebar.error("Gmail情報をすべて入力してください")
    else:
        with st.sidebar:
            with st.spinner("送信中..."):
                ok, msg = send_email(
                    gmail_user, gmail_pass, notify_to,
                    "⬡ Gold Momentum テスト送信",
                    build_email_body(ticker, "STRONG_SELL", 5182.80, 72.3, -3.28, datetime.now().strftime("%Y-%m-%d"))
                )
            if ok:
                st.success("✅ テスト送信成功！")
            else:
                st.error(f"❌ {msg}")

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(t, p):
    df = yf.download(t, period=p, interval="1d", progress=False, auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

if "df" not in st.session_state or fetch_btn:
    with st.spinner(f"{ticker} のデータを取得中..."):
        raw = fetch_data(ticker, period)
    if raw is None or raw.empty:
        st.error(f"❌ {ticker} のデータ取得に失敗しました")
        st.stop()
    st.session_state.df         = raw
    st.session_state.ticker     = ticker
    st.session_state.period     = period_label
    st.session_state.fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.last_notified_date = st.session_state.get("last_notified_date", None)

df_raw = st.session_state.df
df     = generate_signals(df_raw)

# ── Auto notify on latest signal ──────────────────────────────────────────────
latest_sig_row = df[df["signal"].notna()].iloc[-1] if df["signal"].notna().any() else None

if (fetch_btn and latest_sig_row is not None
        and gmail_user and gmail_pass and notify_to
        and notify_map.get(latest_sig_row["signal"], False)):

    latest_date = str(latest_sig_row.name.date())
    if latest_date != st.session_state.get("last_notified_date"):
        sig  = latest_sig_row["signal"]
        ok, msg = send_email(
            gmail_user, gmail_pass, notify_to,
            f"⬡ {SIG[sig]['label']} — {ticker} ${latest_sig_row['Close']:.2f}",
            build_email_body(
                ticker, sig,
                latest_sig_row["Close"],
                latest_sig_row["rsi"],
                latest_sig_row["hist"],
                latest_date
            )
        )
        st.session_state.last_notified_date = latest_date
        if ok:
            st.toast(f"📧 メール送信完了: {SIG[sig]['label']}", icon="✅")
        else:
            st.toast(f"📧 送信失敗: {msg}", icon="❌")

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
    st.metric(f"{st.session_state.ticker} 現在値", f"${latest['Close']:.2f}", f"{day_chg:+.2f}% 前日比")
with k2:
    st.metric("期間リターン", f"{prd_chg:+.2f}%", f"${first['Close']:.2f} → ${latest['Close']:.2f}")
with k3:
    rsi_val    = latest["rsi"]
    rsi_status = "⚠ 買われ過ぎ" if rsi_val > rsi_ob else "⚠ 売られ過ぎ" if rsi_val < rsi_os else "中立域"
    st.metric("RSI (14日)", f"{rsi_val:.1f}", rsi_status)
with k4:
    hist_val = latest["hist"]
    st.metric("MACDヒストグラム", f"{hist_val:.3f}", "上昇 ↑" if hist_val > 0 else "下降 ↓")
with k5:
    if last_sig is not None:
        st.metric("最新シグナル", SIG[last_sig["signal"]]["label"], str(last_sig.name.date()))
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

fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="終値",
    line=dict(color="#FFD700", width=2.5)), row=1, col=1)

for sig_key, info in SIG.items():
    sub = df[df["signal"] == sig_key]
    if sub.empty: continue
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["Close"], mode="markers", name=info["label"],
        marker=dict(color=info["color"], size=12,
                    symbol="triangle-down" if info["is_sell"] else "triangle-up",
                    line=dict(color=CHART_BG, width=1)),
        hovertemplate=f"<b>{info['label']}</b><br>%{{x|%Y-%m-%d}}<br>${{%y:.2f}}<extra></extra>"
    ), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI",
    line=dict(color="#A78BFA", width=1.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[rsi_ob]*len(df), mode="lines", showlegend=False,
    line=dict(color="#FF2D55", width=1, dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[rsi_os]*len(df), mode="lines", showlegend=False,
    line=dict(color="#00E5A0", width=1, dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[50]*len(df), mode="lines", showlegend=False,
    line=dict(color="#1e2535", width=1, dash="dot")), row=2, col=1)

# MACD
colors = ["#00E5A0" if v >= 0 else "#FF2D55" for v in df["hist"].fillna(0)]
fig.add_trace(go.Bar(x=df.index, y=df["hist"], name="ヒストグラム",
    marker_color=colors, opacity=0.65), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD",
    line=dict(color="#00C4FF", width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"], mode="lines", name="Signal",
    line=dict(color="#FFD700", width=1, dash="dash")), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode="lines", showlegend=False,
    line=dict(color="#1e2535", width=1)), row=3, col=1)

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
    st.info("シグナルなし（35日以上のデータが必要）")
else:
    rows = []
    for idx, row in sig_rows.iterrows():
        info = SIG[row["signal"]]
        rows.append({
            "日付":       str(idx.date()),
            "シグナル":   info["label"],
            "価格":       f"${row['Close']:.2f}",
            "RSI":        f"{row['rsi']:.1f}" if pd.notna(row["rsi"]) else "—",
            "MACDヒスト": f"{row['hist']:.3f}" if pd.notna(row["hist"]) else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
