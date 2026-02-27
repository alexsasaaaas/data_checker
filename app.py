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

# ======== 全域參數 =========================================================
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

# -----------------------------------------------------------------------------
# 0. 通用函式與 JSON 序列化
# -----------------------------------------------------------------------------
def json_serial(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Type {type(obj)} not serializable")

# -----------------------------------------------------------------------------
# 1. 週期估計函式（FFT）
# -----------------------------------------------------------------------------
def estimate_period_fft(series: pd.Series) -> int:
    x = series.dropna().values
    if len(x) < 3:
        return 2
    x_centered = x - np.mean(x)
    yf = rfft(x_centered)
    xf = rfftfreq(len(x), d=1)
    idx = np.argmax(np.abs(yf[1:])) + 1   
    freq = xf[idx]
    period = int(round(1 / freq)) if freq > 0 else 2
    return max(period, 2)

# -----------------------------------------------------------------------------
# 2. 子函式：將序列均勻取樣到 THIN_OUT 長度
# -----------------------------------------------------------------------------
def thin(series: pd.Series) -> pd.Series:
    if len(series) <= THIN_OUT:
        return series
    step = int(np.ceil(len(series) / THIN_OUT))
    return series[::step]

# -----------------------------------------------------------------------------
# 3. 六大檢測項目函式
# -----------------------------------------------------------------------------
def classify_dataset(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    seasonality_period: int | None = None,
) -> dict:
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
    df2 = df2.sort_values(time_col)

    # 時間序列 (完整)
    series_full = pd.to_numeric(df2[target], errors="coerce").interpolate()
    series_full.index = df2[time_col]

    # ---- 若過長，先抽樣到 MAX_SERIES_FOR_STL ----
    if len(series_full) > MAX_SERIES_FOR_STL:
        step = int(np.ceil(len(series_full) / MAX_SERIES_FOR_STL))
        series = series_full[::step]
    else:
        series = series_full

    n = len(series)
    period = seasonality_period or estimate_period_fft(series)

    # ---------- ① 週期性分析 --------------------------------------------------
    target_series = series.dropna().values
    if len(target_series) < 4:
        stl = None
        strength = 0.0
    else:
        fft_vals  = np.fft.fft(target_series - np.mean(target_series))
        fft_freqs = np.fft.fftfreq(len(target_series), d=1)
        pos_freqs = fft_freqs[1 : len(target_series)//2]

        if len(pos_freqs) == 0:
            stl = None
            strength = 0.0
        else:
            pos_magnitude = np.abs(fft_vals)[1 : len(target_series)//2]
            top_idx       = np.argsort(pos_magnitude)[-TOP_K_PERIODS:][::-1]
            top_periods   = 1 / pos_freqs[top_idx]

            stl = None
            strength = 0.0
            for period_val in top_periods:
                period_int = int(round(period_val))
                if period_int < 2 or period_int > MAX_STL_PERIOD:
                    continue
                if len(target_series) < 2 * period_int:
                    continue
                if len(target_series) > MAX_STL_N_SAMPLES:
                    step = int(np.ceil(len(target_series) / MAX_STL_N_SAMPLES))
                    ts_for_stl = target_series[::step]
                else:
                    ts_for_stl = target_series

                try:
                    r = STL(ts_for_stl, period=period_int).fit()
                    var_seasonal = np.var(r.seasonal)
                    var_total    = np.var(r.observed)
                    if var_total == 0:
                        continue
                    strength_of_seasonality = var_seasonal / var_total
                    if strength_of_seasonality >= CONFIG["SEASONALITY_THRESHOLD"]:
                        stl      = r
                        strength = strength_of_seasonality
                        break
                except Exception:
                    continue

    # ---------- ② 稀疏率 ------------------------------------------------------
    t0, t1 = series.index.min(), series.index.max()
    intervals = df2[time_col].diff().dt.total_seconds().dropna()
    min_int = int(intervals.min()) if not intervals.empty else None

    if min_int and min_int > 0:
        freq = pd.Timedelta(seconds=min_int)
        full_index = pd.date_range(start=t0, end=t1, freq=freq)
        series_full_span = series.reindex(full_index, fill_value=0)
    else:
        series_full_span = series.copy().fillna(0)

    n_full  = len(series_full_span)
    sparsity = (series_full_span == 0).sum() / n_full

    # ---------- ③ 突發率 ------------------------------------------------------
    diffs = series.diff().abs().dropna()
    thr = diffs.mean() * 3 if not diffs.empty else 0.0
    burst_rate = (diffs > thr).sum() / len(diffs) if not diffs.empty else 0.0

    # ---------- ④ 取樣間隔 ----------------------------------------------------
    intervals = df2[time_col].diff().dt.total_seconds().dropna()
    mode_int = float(intervals.mode().iloc[0]) if not intervals.empty else np.nan
    exp_int: int | None = int(mode_int) if not np.isnan(mode_int) else None

    # ---------- ⑤ 變異數穩定性 ------------------------------------------------
    roll_var = series.rolling(window=period, min_periods=1).var()
    cv = (roll_var.max() - roll_var.min()) / (roll_var.mean() or 1)

    # ---------- ⑥ ADF 平穩性 --------------------------------------------------
    try:
        p_adf = adfuller(series.dropna())[1]
    except Exception:
        p_adf = 1.0

    # ---------- 彙整結果 ------------------------------------------------------
    result = {
        "seasonality_strength": round(strength, 3),
        "seasonality_flag": strength >= CONFIG["SEASONALITY_THRESHOLD"],
        "sparsity_pct": round(sparsity, 3),
        "sparsity_flag": sparsity >= CONFIG["SPARSITY_THRESHOLD"],
        "burst_rate": round(burst_rate, 3),
        "burst_flag": burst_rate >= CONFIG["BURST_THRESHOLD"],
        "mode_interval_sec": exp_int,
        "high_frequency_flag": exp_int is not None and exp_int <= CONFIG["HIGH_FREQ_THRESHOLD"],
        "variance_stability_cv": round(cv, 3),
        "variance_stability_flag": cv < CONFIG["VARIANCE_STABILITY_THRESHOLD"],
        "adf_pvalue": round(p_adf, 3),
        "stationarity_flag": p_adf < CONFIG["STATIONARITY_THRESHOLD"],
    }

    # ---------- 標籤摘要 ------------------------------------------------------
    labels = []
    if result["seasonality_flag"]: labels.append("🌀 Seasonality")
    if result["sparsity_flag"]: labels.append("🌱 Sparsity")
    if result["burst_flag"]: labels.append("⚡ Burst")
    if result["high_frequency_flag"]: labels.append("🔁 High Frequency")
    if result["variance_stability_flag"]: labels.append("🧊 Variance Stability")
    if result["stationarity_flag"]: labels.append("🪵 Stationarity")
    result["labels"] = labels or ["None"]

    # ---------- 中介資料 ------------------------------------------------------
    result.update(
        {
            "__stl_result": stl,
            "__series": thin(series_full),
            "__diffs": thin(diffs),
            "__intervals": thin(intervals),
            "__roll_var": thin(roll_var),
            "__rolling_mean": thin(series.rolling(window=period, min_periods=1).mean()),
            "__rolling_std": thin(series.rolling(window=period, min_periods=1).std()),
        }
    )
    return result

# -----------------------------------------------------------------------------
# 4. Plotly 視覺化函式
# -----------------------------------------------------------------------------
def plot_seasonality(series: pd.Series, stl_result):
    st.subheader("🌀 週期性分析")
    if series.dropna().shape[0] < 3:
        st.info("📉 資料點不足 (<3)，無法繪製週期性分析")
        return
    acf_vals = acf(series.dropna(), nlags=min(40, len(series) - 1))
    fig = go.Figure(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig.update_layout(title="ACF Autocorrelation", xaxis_title="Lag", yaxis_title="ACF")
    st.plotly_chart(fig, use_container_width=True)

    if stl_result is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series.index, y=stl_result.observed, name="Observed"))
        fig2.add_trace(go.Scatter(x=series.index, y=stl_result.trend, name="Trend"))
        fig2.add_trace(go.Scatter(x=series.index, y=stl_result.seasonal, name="Seasonal"))
        fig2.update_layout(title="STL Decomposition", xaxis_title="Time")
        st.plotly_chart(fig2, use_container_width=True)

def plot_sparsity(series: pd.Series):
    st.subheader("🌱 稀疏率")
    if series.dropna().empty:
        st.info("📉 資料不足，無法繪製稀疏率圖")
        return
    zero_mask = series == 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Value"))
    fig.add_trace(
        go.Scatter(
            x=series.index[zero_mask],
            y=[0] * zero_mask.sum(),
            mode="markers",
            marker_symbol="x",
            marker_size=8,
            name="Zero / NA",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Target")
    st.plotly_chart(fig, use_container_width=True)

def plot_burst(series: pd.Series):
    st.subheader("⚡ 突發值分佈 (|Δ|)")
    diffs = series.diff().abs().dropna()
    if diffs.empty:
        st.info("📉 無突發差異值，無法繪製分佈圖")
        return
    fig = px.histogram(diffs, nbins=40)
    fig.update_layout(xaxis_title="|Δ value|", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

def plot_frequency(intervals: pd.Series, mode_sec: int | None, high_freq_flag: bool):
    st.subheader("🔁 取樣間隔 (秒)")
    if intervals.empty or len(intervals) < 2:
        st.info("資料不足，無法計算取樣間隔。")
        return
    fig = px.histogram(intervals, nbins=min(30, len(intervals)))
    fig.update_layout(xaxis_title="Interval (sec)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    if mode_sec is not None:
        st.markdown(f"**主要取樣間隔 (mode)**：≈ **{mode_sec}** 秒；判定：{'✅ 高頻' if high_freq_flag else '❌ 非高頻'}")

def plot_variance_stability(roll_var: pd.Series):
    st.subheader("🧊 滾動變異數")
    if roll_var.dropna().empty:
        st.info("📉 資料不足，無法繪製變異數穩定性圖")
        return
    fig = go.Figure(go.Scatter(x=roll_var.index, y=roll_var, mode="lines"))
    fig.update_layout(xaxis_title="Time", yaxis_title="Rolling Variance")
    st.plotly_chart(fig, use_container_width=True)

def plot_stationarity(series: pd.Series, rolling_mean: pd.Series, rolling_std: pd.Series):
    st.subheader("🪵 平穩性檢測 (Rolling Mean / Std)")
    if series.dropna().empty:
        st.info("📉 資料不足，無法繪製平穩性檢測圖")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="原始序列", opacity=0.4))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name="滾動平均"))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, name="滾動標準差"))
    fig.update_layout(xaxis_title="時間", yaxis_title="數值", title="平穩性檢測圖")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 工具函式：自動偵測時間欄位
# -----------------------------------------------------------------------------
def detect_time_column(df: pd.DataFrame) -> str | None:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                candidates.append(col)
            except Exception:
                pass
    if len(candidates) == 1:
        return candidates[0]
    for col in candidates:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return candidates[0] if candidates else None

# -----------------------------------------------------------------------------
# 6. 主介面
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="時間序列特性檢測", layout="wide")
    st.title("⏱️ 時間序列六大特性檢測")

    uploaded = st.file_uploader("請上傳包含時間與數值欄位的 CSV 檔", type=["csv"])
    if uploaded is None:
        st.info("等待上傳 CSV 檔案…")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()

        if df.empty:
            st.error("上傳的檔案是空的")
            st.stop()

        if len(df.columns) < 2:
            st.error("檔案必須至少包含兩個欄位")
            st.stop()

        dataset_name = uploaded.name
        auto_time = detect_time_column(df)
        time_col = st.sidebar.selectbox(
            "選擇時間欄位",
            ([auto_time] if auto_time else []) + list(df.columns)
        )

        if not time_col:
            st.error("請選擇時間欄位")
            st.stop()

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        valid_ratio = df[time_col].notna().mean()
        if valid_ratio < 0.5:
            st.error(f"欄位「{time_col}」轉成時間的有效值不足 ({valid_ratio:.0%})，請重新選擇。")
            st.stop()

        candidate_targets = [c for c in df.columns if c != time_col]
        target_col = st.sidebar.selectbox(
              "選擇目標欄位（將會嘗試轉為 numeric）",
        candidate_targets
        )

        clean_target = (
            df[target_col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .str.replace(",", "")
            .replace("", np.nan)
        )
        df[target_col] = pd.to_numeric(clean_target, errors="coerce")

        if df[target_col].isna().all():
            st.error(f"欄位「{target_col}」無有效數值，請檢查原始資料")
            st.stop()

        if len(df) < 4:
            st.error("數據樣本數量不足，至少需要 4 個樣本")
            st.stop()

        group_cols = st.sidebar.multiselect(
            "多重時間序列 (最多3欄)", options=[c for c in df.columns if c not in (time_col, target_col)]
        )
        if len(group_cols) > 3:
            st.sidebar.error("最多只能選擇 3 個分組欄位")
            st.stop()

        if group_cols:
            unique_count = df.dropna(subset=group_cols).drop_duplicates(subset=group_cols).shape[0]
            st.info(f"將使用 {unique_count} 組時間序列進行建模")
            res_list = []
            for keys, subdf in df.groupby(group_cols):
                res = classify_dataset(subdf, time_col, target_col)
                if isinstance(keys, tuple):
                    for i, col in enumerate(group_cols):
                        res[col] = keys[i]
                else:
                    res[group_cols[0]] = keys
                res_list.append(res)
            
            df_metrics_group = pd.DataFrame(res_list)
            metrics_cols = [
                'seasonality_strength','sparsity_pct','burst_rate',
                'mode_interval_sec','variance_stability_cv','adf_pvalue',
                'seasonality_flag','sparsity_flag','burst_flag',
                'high_frequency_flag','variance_stability_flag','stationarity_flag','labels'
            ]
            display_cols = group_cols + metrics_cols
            df_metrics_group = df_metrics_group[display_cols]
            st.subheader("📊 分組檢測結果")
            st.table(df_metrics_group)
            st.download_button(
                "⬇️ 下載 JSON 報告",
                data=json.dumps(res_list, default=json_serial, ensure_ascii=False, indent=2),
                file_name="ts_features_report.json",
                mime="application/json",
            )
            st.stop()

        res = classify_dataset(df, time_col, target_col)

        st.subheader(f"📊 檢測結果：{dataset_name}")
        df_metrics = pd.DataFrame(
            [
                {"特性": "週期性 strength", "值": res["seasonality_strength"], "判定": res["seasonality_flag"]},
                {"特性": "稀疏率 sparsity_pct", "值": res["sparsity_pct"], "判定": res["sparsity_flag"]},
                {"特性": "突發率 burst_rate", "值": res["burst_rate"], "判定": res["burst_flag"]},
                {"特性": "高頻 interval(s)", "值": res["mode_interval_sec"] if res["mode_interval_sec"] is not None else "—", "判定": res["high_frequency_flag"]},
                {"特性": "變異數 CV", "值": res["variance_stability_cv"], "判定": res["variance_stability_flag"]},
                {"特性": "平穩性 ADF p-value", "值": res["adf_pvalue"], "判定": res["stationarity_flag"]},
            ]
        )
        st.table(df_metrics)
        summary = "、".join(res["labels"])
        st.markdown(f"**🔍 總體判定：** 這份資料集可能屬於 {summary}")

        plot_seasonality(res["__series"], res["__stl_result"])
        plot_sparsity(res["__series"])
        plot_burst(res["__series"])
        plot_frequency(res["__intervals"], res["mode_interval_sec"], res["high_frequency_flag"])
        plot_variance_stability(res["__roll_var"])
        plot_stationarity(res["__series"], res["__rolling_mean"], res["__rolling_std"])

        res_download = {k: v for k, v in res.items() if not k.startswith("__")}
        st.download_button(
            "⬇️ 下載 JSON 報告",
            data=json.dumps(res_download, default=json_serial, ensure_ascii=False, indent=2),
            file_name="ts_features_report.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"處理檔案時發生錯誤: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()