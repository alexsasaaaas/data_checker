import sys
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from numpy.fft import rfft, rfftfreq
import plotly.graph_objects as go
import plotly.express as px
import json

# ======== 全域參數 ========
MAX_STL_PERIOD       = 365
MAX_SERIES_FOR_STL   = 20_000
MAX_STL_N_SAMPLES    = 10_000
TOP_K_PERIODS        = 3
THIN_OUT             = 2_000

CONFIG = {
    "SEASONALITY_THRESHOLD": 0.3,
    "SPARSITY_THRESHOLD": 0.5,
    "BURST_THRESHOLD": 0.1,
    "HIGH_FREQ_THRESHOLD": 60,
    "VARIANCE_STABILITY_THRESHOLD": 0.2,
    "STATIONARITY_THRESHOLD": 0.05,
}

# --- 工具函式 ---
def json_serial(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)

def estimate_period_fft(series: pd.Series) -> int:
    x = series.dropna().values
    if len(x) < 10: return 2 # 樣本太少不預測週期
    x_centered = x - np.mean(x)
    yf = rfft(x_centered)
    xf = rfftfreq(len(x), d=1)
    idx = np.argmax(np.abs(yf[1:])) + 1
    period = int(round(1 / xf[idx])) if xf[idx] > 0 else 2
    return max(period, 2)

def thin(series: pd.Series) -> pd.Series:
    if len(series) <= THIN_OUT: return series
    step = int(np.ceil(len(series) / THIN_OUT))
    return series[::step]

# --- 核心分析邏輯 ---
def classify_dataset(df, time_col, target, seasonality_period=None):
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
    df2 = df2.dropna(subset=[time_col]).sort_values(time_col)

    series_full = pd.to_numeric(df2[target], errors="coerce").interpolate()
    series_full.index = df2[time_col]

    # 取樣以進行運算
    series = series_full[::int(np.ceil(len(series_full)/MAX_SERIES_FOR_STL))] if len(series_full) > MAX_SERIES_FOR_STL else series_full
    n = len(series)
    
    # 1. 週期性 (FFT + STL)
    target_series = series.dropna().values
    stl, strength = None, 0.0
    if len(target_series) >= 4:
        fft_vals = np.fft.fft(target_series - np.mean(target_series))
        fft_freqs = np.fft.fftfreq(len(target_series))
        pos_freqs = fft_freqs[1:len(target_series)//2]
        if len(pos_freqs) > 0:
            pos_mag = np.abs(fft_vals)[1:len(target_series)//2]
            top_periods = 1 / pos_freqs[np.argsort(pos_mag)[-TOP_K_PERIODS:][::-1]]
            for p_val in top_periods:
                p_int = int(round(p_val))
                if 2 <= p_int <= MAX_STL_PERIOD and len(target_series) >= 2*p_int:
                    try:
                        r = STL(target_series, period=p_int).fit()
                        s_str = np.var(r.seasonal) / np.var(r.observed) if np.var(r.observed) > 0 else 0
                        if s_str >= CONFIG["SEASONALITY_THRESHOLD"]:
                            stl, strength = r, s_str
                            break
                    except: continue

    # 2. 稀疏率
    sparsity = ((series == 0) | series.isna()).sum() / n
    # 3. 突發率
    diffs = series.diff().abs().dropna()
    burst_rate = (diffs > diffs.mean()*3).sum()/len(diffs) if not diffs.empty else 0
    # 4. 間隔
    intervals = df2[time_col].diff().dt.total_seconds().dropna()
    mode_int = float(intervals.mode().iloc[0]) if not intervals.empty else None
    # 5. 變異數穩定性
    period_for_roll = int(round(top_periods[0])) if 'top_periods' in locals() else 10
    roll_var = series.rolling(window=max(period_for_roll, 2), min_periods=1).var()
    cv = (roll_var.max() - roll_var.min()) / (roll_var.mean() or 1)
    # 6. 平穩性
    try: p_adf = adfuller(series.dropna())[1]
    except: p_adf = 1.0

    res = {
        "seasonality_strength": round(strength, 3),
        "seasonality_flag": strength >= CONFIG["SEASONALITY_THRESHOLD"],
        "sparsity_pct": round(sparsity, 3),
        "sparsity_flag": sparsity >= CONFIG["SPARSITY_THRESHOLD"],
        "burst_rate": round(burst_rate, 3),
        "burst_flag": burst_rate >= CONFIG["BURST_THRESHOLD"],
        "mode_interval_sec": mode_int,
        "high_frequency_flag": mode_int is not None and mode_int <= CONFIG["HIGH_FREQ_THRESHOLD"],
        "variance_stability_cv": round(cv, 3),
        "variance_stability_flag": cv < CONFIG["VARIANCE_STABILITY_THRESHOLD"],
        "adf_pvalue": round(p_adf, 3),
        "stationarity_flag": p_adf < CONFIG["STATIONARITY_THRESHOLD"],
    }
    
    labels = []
    if res["seasonality_flag"]: labels.append("🌀 Seasonality")
    if res["sparsity_flag"]: labels.append("🌱 Sparsity")
    if res["burst_flag"]: labels.append("⚡ Burst")
    if res["high_frequency_flag"]: labels.append("🔁 High Frequency")
    if res["variance_stability_flag"]: labels.append("🧊 Variance Stability")
    if res["stationarity_flag"]: labels.append("🪵 Stationarity")
    res["labels"] = labels or ["None"]
    
    # 暫存繪圖數據
    res.update({"__stl_result": stl, "__series": thin(series_full), "__intervals": thin(intervals), 
                "__roll_var": thin(roll_var), "__rolling_mean": thin(series.rolling(10).mean())})
    return res

# --- UI 介面 ---
def main():
    st.set_page_config(page_title="TS Feature Detector", layout="wide")
    st.title("⏱️ 時間序列六大特性檢測")

    uploaded = st.file_uploader("上傳 CSV 檔案", type=["csv"])
    if not uploaded:
        st.info("請上傳檔案以開始分析")
        return

    try:
        df = pd.read_csv(uploaded)
        col1, col2 = st.sidebar.columns(2)
        
        # 自動偵測時間欄位
        time_col = st.sidebar.selectbox("時間欄位", df.columns)
        target_col = st.sidebar.selectbox("數值欄位", [c for c in df.columns if c != time_col])
        
        if st.sidebar.button("開始執行分析"):
            with st.spinner('分析中...'):
                res = classify_dataset(df, time_col, target_col)
                
                # 顯示表格
                st.subheader("📊 檢測結果")
                display_df = pd.DataFrame([
                    {"特性": k, "數值": v} for k, v in res.items() if not k.startswith("__") and k != 'labels'
                ])
                st.table(display_df)
                st.success(f"總體判定：{', '.join(res['labels'])}")

                # 繪圖
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.line(res["__series"], title="原始序列與趨勢")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if res["__stl_result"] is not None:
                        st.write("STL 週期性分解已完成")
                        
                st.download_button("下載 JSON 報告", 
                                   data=json.dumps({k:v for k,v in res.items() if not k.startswith("__")}, default=json_serial),
                                   file_name="report.json")
    except Exception as e:
        st.error(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()