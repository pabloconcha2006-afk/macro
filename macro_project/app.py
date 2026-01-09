#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 03:26:13 2026

@author: pabloconcha
"""

# -*- coding: utf-8 -*-
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3
import yfinance as yf
from io import StringIO
from pandas.tseries.offsets import MonthEnd, QuarterEnd

# ---- pypfopt (puede faltar en Streamlit Cloud si no está en requirements.txt) ----
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
except ModuleNotFoundError:
    st.error("Falta 'pypfopt'. Agrega 'PyPortfolioOpt' a requirements.txt y vuelve a desplegar.")
    st.stop()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# CONFIG (TU LISTA SP500_ALL SE QUEDA IGUAL)
# =========================
SP500_ALL = ["A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES",
"AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMD", "AME", "AMGN", "AMP", "AMT",
"AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK",
"AXON", "AXP", "AYI", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO",
"BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH",
"CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW",
"CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP",
"COR", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CW", "D",
"DAL", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI",
"DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR",
"ENPH", "EOG", "EPAM", "EQIX", "EQR", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG",
"FAST", "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GILD",
"GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWRE", "GWW", "HAL", "HAS", "HBAN",
"HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM",
"HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTU", "INVH", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ",
"J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB",
"KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX",
"LULU", "LUV", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET",
"META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA",
"MS", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NEE", "NEM", "NFG", "NFLX", "NI", "NKE", "NOC",
"NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON",
"ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR",
"PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PTC",
"PWR", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST",
"RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SIRI", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SPG", "SPGI",
"SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL",
"TER", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
"TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VMC",
"VTRS", "VRTX", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST",
"WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
   
]

STPS_XLS_PATH = "stps_xls/302_0074.xls"  # .xls requiere xlrd instalado en el deploy

BANXICO_TOKEN = "710446e886e30952133c0d23a4882d24fb2aafaa659b416b07a05d7b19d2a10f"
UA = "Mozilla/5.0"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Si no tienes FRED_API_KEY en tu entorno, usa este valor:
FRED_API_KEY = os.getenv("FRED_API_KEY") or "0eddd94e8ad9ae52f0559349ed41ee8f"

# Ventana fija (tu pipeline)
START_MONTH = "2024-02"   # últimos 24 meses vs ene-2026 -> 2024-02 .. 2026-01
END_MONTH   = "2026-01"

# =========================
# HELPERS 24 MESES (BANXICO)
# =========================
def two_full_years_window(today=None):
    today = today or pd.Timestamp.today().normalize()
    end = today.replace(day=1) - pd.Timedelta(days=1)
    start = (end - pd.DateOffset(months=23)).replace(day=1)
    return start, end

def banxico_series_24m(series_id: str, nombre: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = (
        "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
        f"{series_id}/datos/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    )
    r = requests.get(url, headers={"Bmx-Token": BANXICO_TOKEN, "User-Agent": UA}, timeout=25)
    r.raise_for_status()
    datos = r.json()["bmx"]["series"][0]["datos"]

    rows = []
    for d in datos:
        fecha = pd.to_datetime(d.get("fecha"), dayfirst=True, errors="coerce")
        val = pd.to_numeric(str(d.get("dato", "")).replace(",", ""), errors="coerce")
        if pd.notna(fecha) and pd.notna(val):
            rows.append({"Indicador": nombre, "Fecha": fecha, "Valor": float(val)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Fecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("Fecha").groupby(["Fecha", "Indicador"], as_index=False)["Valor"].last()
    df = df[(df["Fecha"] >= start) & (df["Fecha"] <= end)]
    return df
    default_tickers
# =========================
# HELPERS (TIME) PARA OECD/FRED
# =========================
def month_end_index(start_ym: str, end_ym: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_ym + "-01") + MonthEnd(0)
    end   = pd.Timestamp(end_ym + "-01") + MonthEnd(0)
    return pd.date_range(start=start, end=end, freq="ME")

def to_month_end(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(s.index, errors="coerce")
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).dropna()
    s.index = (s.index + MonthEnd(0)).normalize()
    return s.groupby(s.index).last().sort_index()

def to_quarter_end(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(s.index, errors="coerce")
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).dropna()
    s.index = (s.index + QuarterEnd(0)).normalize()
    return s.groupby(s.index).last().sort_index()

def q_to_month_ffill(s_q: pd.Series, month_index: pd.DatetimeIndex) -> pd.Series:
    s_q = to_quarter_end(s_q)
    if s_q.empty:
        return pd.Series(index=month_index, dtype="float64")
    return s_q.reindex(month_index, method="ffill")

# =========================
# HELPERS (FETCH) OECD/FRED
# =========================
def fetch_oecd_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def oecd_series_from_all(
    url_all: str,
    name: str,
    must: dict,
    time_parse: str,     # "monthly" | "quarterly" | "annual"
    score_dims: list,    # columnas-dimensiones para agrupar
    month_index: pd.DatetimeIndex,
    ref_area: str = "MEX",
    ffill_to_month: bool = True
) -> pd.Series:
    try:
        df = fetch_oecd_csv(url_all)
    except Exception:
        return pd.Series(index=month_index, dtype="float64")

    if "REF_AREA" not in df.columns or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.Series(index=month_index, dtype="float64")

    mx = df[df["REF_AREA"] == ref_area].copy()
    if mx.empty:
        return pd.Series(index=month_index, dtype="float64")

    for k, v in (must or {}).items():
        if k in mx.columns:
            mx = mx[mx[k].astype(str) == str(v)]

    if mx.empty:
        return pd.Series(index=month_index, dtype="float64")

    mx["OBS_VALUE"] = pd.to_numeric(mx["OBS_VALUE"], errors="coerce")
    mx = mx.dropna(subset=["OBS_VALUE"])
    if mx.empty:
        return pd.Series(index=month_index, dtype="float64")

    # pick best group by count of non-null
    if score_dims:
        g = mx.groupby(score_dims)["OBS_VALUE"].count().sort_values(ascending=False)
        best_key = g.index[0]
        if not isinstance(best_key, tuple):
            best_key = (best_key,)
        mask = np.ones(len(mx), dtype=bool)
        for col, val in zip(score_dims, best_key):
            mask &= (mx[col].astype(str) == str(val))
        mx = mx[mask].copy()

    t = mx["TIME_PERIOD"].astype(str)

    if time_parse == "monthly":
        idx = pd.to_datetime(t, errors="coerce")
        s = pd.Series(mx["OBS_VALUE"].values, index=idx).dropna().sort_index()
        s = to_month_end(s)
        return s.reindex(month_index)

    if time_parse == "quarterly":
        idx = pd.PeriodIndex(t, freq="Q").to_timestamp(how="end")
        s = pd.Series(mx["OBS_VALUE"].values, index=idx).dropna().sort_index()
        if ffill_to_month:
            return q_to_month_ffill(s, month_index)
        return to_quarter_end(s).reindex(month_index)

    if time_parse == "annual":
        idx = pd.to_datetime(t + "-12-31", errors="coerce")
        s = pd.Series(mx["OBS_VALUE"].values, index=idx).dropna().sort_index()
        s.index = s.index.normalize()
        return s.reindex(month_index, method="ffill")

    return pd.Series(index=month_index, dtype="float64")

def fred_series(series_id: str, observation_start: str = "2024-01-01") -> pd.Series:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        "&file_type=json"
        f"&observation_start={observation_start}"
    )
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    if not obs:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna().sort_index()

def first_nonempty_fred(series_ids: list) -> pd.Series:
    for sid in series_ids:
        try:
            s = fred_series(sid)
            if s.notna().any():
                return s
        except Exception:
            pass
    return pd.Series(dtype="float64")

# =========================
# OECD URLS (ALL) — LOS QUE YA TE FUNCIONARON
# =========================
URL_CLI = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_CLI/.M.LI...AA...H"
    "?startPeriod=2024-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)

URL_CPI = (
    "https://sdmx.oecd.org/public/rest/data/"
    "MEI_CPI/MEX.CPALTT01.GP.M/all"
    "?startPeriod=2024-01&format=csvfilewithlabels"
)

URL_UNEMP_ALL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/all"
    "?startPeriod=2024-01&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
)

URL_LFP_ALL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.TPS,DSD_LFS@DF_IALFS_LF_WAP_Q,1.0/all"
    "?startPeriod=2024-01&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
)

URL_INDSERV_ALL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_INDSERV/all"
    "?startPeriod=2024-01&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
)

URL_CONS_ALL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE5_T117,1.0/all"
    "?startPeriod=2024-01&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
)

# =========================
# BUILD TABLA MACRO (REEMPLAZA INEGI)
# =========================
def build_macro_table_24m() -> pd.DataFrame:
    mi = month_end_index(START_MONTH, END_MONTH)
    out = pd.DataFrame(index=mi)

    # 1) IGAE a/a (CLI nivel) + IGAE m/m (% m/m)
    s_cli = oecd_series_from_all(
        URL_CLI,
        name="CLI",
        must={"FREQ": "M", "MEASURE": "LI", "TRANSFORMATION": "IX"},
        time_parse="monthly",
        score_dims=["FREQ", "MEASURE", "TRANSFORMATION", "ADJUSTMENT", "TIME_HORIZ", "METHODOLOGY"],
        month_index=mi,
    )
    out["IGAE a/a"] = s_cli
    out["IGAE m/m"] = out["IGAE a/a"].pct_change(fill_method=None) * 100

    # 2) Inflación (CPI) — endpoint que ya te dio filas
    try:
        df_cpi = fetch_oecd_csv(URL_CPI)
        mx = df_cpi[df_cpi["REF_AREA"] == "MEX"].copy()
        mx["TIME_PERIOD"] = pd.to_datetime(mx["TIME_PERIOD"], errors="coerce")
        mx["OBS_VALUE"] = pd.to_numeric(mx["OBS_VALUE"], errors="coerce")
        s_inf = mx.dropna(subset=["TIME_PERIOD"]).set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
        out["Inflacion"] = to_month_end(s_inf).reindex(mi)
    except Exception:
        out["Inflacion"] = np.nan

    # 3) Desempleo (OECD mensual)
    out["Desempleo"] = oecd_series_from_all(
        URL_UNEMP_ALL,
        name="Desempleo",
        must={
            "FREQ": "M",
            "MEASURE": "UNE_LF_M",
            "UNIT_MEASURE": "PT_LF_SUB",
            "TRANSFORMATION": "_Z",
            "ADJUSTMENT": "N",
            "SEX": "_T",
            "AGE": "Y_GE15",
            "ACTIVITY": "_Z",
        },
        time_parse="monthly",
        score_dims=["FREQ", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION", "ADJUSTMENT", "SEX", "AGE", "ACTIVITY"],
        month_index=mi,
    )

    # 4) Participación empleo — OECD trimestral -> mensual (ffill)
    out["Partic Empleo"] = oecd_series_from_all(
        URL_LFP_ALL,
        name="Partic Empleo",
        must={
            "MEASURE": "LF_WAP",
            "UNIT_MEASURE": "PT_WAP_SUB",
            "TRANSFORMATION": "_Z",
            "ADJUSTMENT": "N",
            "SEX": "_T",
            "AGE": "Y_GE15",
            "ACTIVITY": "_Z",
            "FREQ": "Q",
        },
        time_parse="quarterly",
        score_dims=["FREQ", "MEASURE", "UNIT_MEASURE", "TRANSFORMATION", "ADJUSTMENT", "SEX", "AGE", "ACTIVITY"],
        month_index=mi,
        ffill_to_month=True,
    )

    # 5) PIB a/a (FRED, trimestral -> mensual ffill)
    s_pib = first_nonempty_fred(["MEXGDPRQPSMEI"])
    out["PIB a/a"] = q_to_month_ffill(s_pib, mi)

    # 6) Act Industrial (OECD INDSERV) — auto-pick
    out["Act Industrial"] = oecd_series_from_all(
        URL_INDSERV_ALL,
        name="Act Industrial",
        must={"FREQ": "M", "UNIT_MEASURE": "IX", "TRANSFORMATION": "_Z", "ADJUSTMENT": "N"},
        time_parse="monthly",
        score_dims=["ACTIVITY", "MEASURE", "ADJUSTMENT", "FREQ", "UNIT_MEASURE", "TRANSFORMATION"],
        month_index=mi,
    )

    # 7) Consumo Bienes / Servicios (OECD anual 2024 -> mensual ffill)
    try:
        df_cons = fetch_oecd_csv(URL_CONS_ALL)
        mx = df_cons[df_cons["REF_AREA"] == "MEX"].copy()
        mx["OBS_VALUE"] = pd.to_numeric(mx["OBS_VALUE"], errors="coerce")
        mx_a = mx[mx["FREQ"].astype(str) == "A"].copy()

        def annual_to_month_ffill(mx_a_sub: pd.DataFrame) -> pd.Series:
            if mx_a_sub.empty:
                return pd.Series(index=mi, dtype="float64")
            idx = pd.to_datetime(mx_a_sub["TIME_PERIOD"].astype(str) + "-12-31", errors="coerce")
            s = pd.Series(mx_a_sub["OBS_VALUE"].values, index=idx).dropna().sort_index()
            s.index = s.index.normalize()
            return s.reindex(mi, method="ffill")

        bienes = mx_a[(mx_a["TRANSACTION"].astype(str) == "P31DC") & (mx_a["UNIT_MEASURE"].astype(str) == "XDC")].copy()
        serv   = mx_a[(mx_a["TRANSACTION"].astype(str) == "P314")  & (mx_a["UNIT_MEASURE"].astype(str) == "XDC")].copy()

        out["Consumo Bienes"] = annual_to_month_ffill(bienes)
        out["Consumo Servicios"] = annual_to_month_ffill(serv)
    except Exception:
        out["Consumo Bienes"] = np.nan
        out["Consumo Servicios"] = np.nan

    # 8) IFB Total / IFB Const (FRED, trimestral -> mensual ffill)
    s_ifb_total = first_nonempty_fred(["NFIRSAXDCMXQ", "NFIRNSAXDCMXQ"])
    s_ifb_const = first_nonempty_fred(["NFIRNSAXDCMXQ", "NFIRSAXDCMXQ"])
    out["IFB Total"] = q_to_month_ffill(s_ifb_total, mi)
    out["IFB Const"] = q_to_month_ffill(s_ifb_const, mi)

    # 9) Ind Manufac (FRED) — fallbacks
    s_man = first_nonempty_fred(["MEXPROMANMISMEI", "MEXPRMNTO01IXOBM", "MEXPROMANQISMEI"])
    out["Ind Manufac"] = to_month_end(s_man).reindex(mi)

    cols = [
        "Consumo Bienes", "Consumo Servicios", "Partic Empleo", "Desempleo",
        "PIB a/a", "IGAE a/a", "IGAE m/m", "Inflacion", "Act Industrial",
        "IFB Const", "IFB Total", "Ind Manufac"
    ]
    out = out[cols]
    out.index = out.index.strftime("%Y-%m")
    return out

# =========================
# ÚLTIMO DATO (TABLA DETALLE) — REEMPLAZA INEGI POR TABLA MACRO
# =========================
def fetch_all_data() -> pd.DataFrame:
    results = []

    # 0) TABLA MACRO (OECD/FRED) — para métricas “tipo INEGI”
    try:
        macro = build_macro_table_24m()
        last_ym = macro.dropna(how="all").index.max()
        last_row = macro.loc[last_ym] if last_ym is not None else pd.Series(dtype=float)
    except Exception as e:
        last_ym = None
        last_row = pd.Series(dtype=float)
        results.append({"Fuente": "Macro (OECD/FRED)", "Indicador": "Macro table", "Valor": 0.0, "Fecha": f"Error: {e}"})

    def add_macro(ind_name, col_name, fuente):
        if col_name in last_row.index and pd.notna(last_row[col_name]):
            results.append({"Fuente": fuente, "Indicador": ind_name, "Valor": float(last_row[col_name]), "Fecha": str(last_ym)})
        else:
            results.append({"Fuente": fuente, "Indicador": ind_name, "Valor": np.nan, "Fecha": str(last_ym)})

    add_macro("Inflación", "Inflacion", "OECD")
    add_macro("Tasa Desempleo", "Desempleo", "OECD")
    add_macro("IGAE", "IGAE a/a", "OECD")
    add_macro("PIB a/a", "PIB a/a", "FRED")

    # 1) STPS local
    try:
        df_stps = pd.read_excel(STPS_XLS_PATH, sheet_name="Salario Mínimo 2019-2025", engine="xlrd")
        val_salario = pd.to_numeric(df_stps.stack(), errors="coerce").dropna().iloc[-1]
        results.append(
            {"Fuente": "STPS", "Indicador": "Salario Mínimo", "Valor": float(val_salario), "Fecha": datetime.today().strftime("%Y-%m-%d")}
        )
    except Exception as e:
        results.append({"Fuente": "STPS", "Indicador": "Salario Mínimo", "Valor": np.nan, "Fecha": f"Error: {e}"})

    # 2) Banxico oportuno
    bx_last = {
        "Exportaciones": "SE36664",
        "Importaciones": "SE36672",
        "Balanza Petrolera": "SE36683",
        "Cetes 28d": "SF43936",
    }
    for nom, sid in bx_last.items():
        try:
            r = requests.get(
                f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{sid}/datos/oportuno",
                headers={"Bmx-Token": BANXICO_TOKEN, "User-Agent": UA},
                timeout=20,
            )
            r.raise_for_status()
            d = r.json()["bmx"]["series"][0]["datos"][0]
            results.append(
                {"Fuente": "Banxico", "Indicador": nom, "Valor": float(str(d["dato"]).replace(",", "")), "Fecha": d["fecha"]}
            )
        except Exception as e:
            results.append({"Fuente": "Banxico", "Indicador": nom, "Valor": np.nan, "Fecha": f"Error: {e}"})

    return pd.DataFrame(results)

# =========================
# MIN VAR
# =========================
def get_min_volatility_portfolio(tickers):
    if len(tickers) < 2:
        return None, None
    try:
        data = yf.download(tickers, period="3y", progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            precios = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
        else:
            precios = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

        if isinstance(precios, pd.DataFrame):
            precios = precios.dropna(how="all")

        mu = expected_returns.mean_historical_return(precios)
        S = risk_models.sample_cov(precios)

        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance()
        return cleaned_weights, perf
    except Exception as e:
        st.error(f"Error interno en la optimización: {e}")
        return None, None

# =========================
# MEJOR PORTAFOLIO (max_sharpe sobre combinaciones)
# =========================
def download_prices(tickers, period="3y", min_obs=252 * 2, chunk_size=100):
    closes = []
    tickers = list(dict.fromkeys(tickers))

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        data = yf.download(chunk, period=period, progress=False, auto_adjust=True, group_by="column")
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                continue
            px = data["Close"]
        else:
            if "Close" in data.columns and len(chunk) == 1:
                px = data[["Close"]].rename(columns={"Close": chunk[0]})
            else:
                continue

        closes.append(px)

    if not closes:
        return pd.DataFrame()

    px = pd.concat(closes, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]

    ok = px.count()
    ok = ok[ok >= min_obs].index.tolist()
    px = px[ok]

    px = px.sort_index().ffill(limit=5)
    px = px.dropna(how="all")
    return px

def preselect_universe(prices: pd.DataFrame, m=80, top_sharpe=200, rf=0.0):
    if prices is None or prices.empty or prices.shape[1] < 2:
        return []

    rets = prices.pct_change()

    min_ret_obs = int(0.80 * rets.shape[0])
    ok_cols = rets.count()
    ok_cols = ok_cols[ok_cols >= min_ret_obs].index.tolist()
    rets = rets[ok_cols].dropna(how="all")

    rets = rets.dropna(how="any")
    if rets.empty or rets.shape[1] < 2:
        return []

    mu_i = rets.mean() * 252
    vol_i = rets.std(ddof=1) * np.sqrt(252)
    sharpe_i = (mu_i - rf) / vol_i
    sharpe_i = sharpe_i.replace([np.inf, -np.inf], np.nan).dropna()
    if sharpe_i.empty:
        return []

    cand = sharpe_i.sort_values(ascending=False).head(min(top_sharpe, len(sharpe_i))).index.tolist()
    if len(cand) < 2:
        return cand

    corr = rets[cand].corr().fillna(1.0)
    selected = [cand[0]]

    while len(selected) < min(m, len(cand)):
        remaining = [t for t in cand if t not in selected]
        avg_corr = corr.loc[remaining, selected].mean(axis=1)
        next_t = avg_corr.sort_values().index[0]
        selected.append(next_t)

    return selected

def max_sharpe_for_set(prices: pd.DataFrame, rf=0.0):
    prices = prices.sort_index().ffill(limit=5).dropna(how="any")

    if prices.shape[0] < 200 or prices.shape[1] < 2:
        raise ValueError("Datos insuficientes")

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=rf)

    w = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rf)
    return sharpe, (ret, vol), w

def search_best_combo_random(prices: pd.DataFrame, tickers, k=5, trials=50000, rf=0.0, seed=123):
    rng = np.random.default_rng(seed)
    best = {"sharpe": -np.inf, "combo": None, "perf": None, "weights": None}

    tickers = [t for t in tickers if t in prices.columns]
    if len(tickers) < k:
        return best

    for _ in range(trials):
        combo = tuple(rng.choice(tickers, size=k, replace=False))
        px = prices[list(combo)]
        try:
            sharpe, perf, w = max_sharpe_for_set(px, rf=rf)
            if sharpe > best["sharpe"]:
                best = {"sharpe": sharpe, "combo": combo, "perf": perf, "weights": w}
        except Exception:
            continue

    return best

def option3_best_sp500_combo(
    sp500_all,
    k=5,
    period="3y",
    rf=0.0,
    min_obs=252,
    m=80,
    top_sharpe=200,
    trials=50000,
    seed=123,
):
    prices_all = download_prices(sp500_all, period=period, min_obs=min_obs)
    preselected = preselect_universe(prices_all, m=m, top_sharpe=top_sharpe, rf=rf)

    if len(preselected) < k:
        return {"sharpe": -np.inf, "combo": None, "perf": None, "weights": None, "weights_series": pd.Series(dtype=float)}

    best = search_best_combo_random(prices_all[preselected], preselected, k=k, trials=trials, rf=rf, seed=seed)

    if best["combo"] is None:
        best["weights_series"] = pd.Series(dtype=float)
        return best

    w = pd.Series(best["weights"], dtype=float)
    w = w[w > 0].sort_values(ascending=False)
    best["weights_series"] = w
    return best

# =========================
# UI
# =========================
st.set_page_config(page_title="Macro Mexico Pro", layout="wide")

st.markdown(
    """
<style>
.stMetric { background-color: #0e1117; border: 1px solid #262730; padding: 15px; border-radius: 10px; }
[data-testid="stMetricValue"] { color: #00d4ff; font-size: 32px; font-weight: bold; }
.main { background-color: #050505; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Monitor Económico: México")
st.caption("STPS (archivo), Banxico (API), OECD/FRED (reemplazo de INEGI)")

with st.spinner("Sincronizando..."):
    df = fetch_all_data()

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)

    def get_val(nombre):
        temp = df[df["Indicador"].astype(str).str.contains(nombre, case=False, na=False)]
        v = temp.iloc[0]["Valor"] if not temp.empty else np.nan
        return float(v) if pd.notna(v) else np.nan

    c1.metric("💰 Salario Mínimo", f"${(get_val('Salario') or 0.0):,.2f}")
    c2.metric("🏷️ Inflación", f"{(get_val('Inflación') if pd.notna(get_val('Inflación')) else 0.0):,.2f}%")
    c3.metric("📉 Cetes 28d", f"{(get_val('Cetes') if pd.notna(get_val('Cetes')) else 0.0):,.2f}%")
    c4.metric("🏗️ IGAE", f"{(get_val('IGAE') if pd.notna(get_val('IGAE')) else 0.0):,.2f}")

st.divider()
st.subheader("📋 Detalle de Indicadores (último dato)")

df_vis = df.copy()

def fmt_val(row):
    ind = str(row.get("Indicador", ""))
    v = pd.to_numeric(row.get("Valor"), errors="coerce")
    if pd.isna(v):
        return str(row.get("Valor"))
    if any(k in ind for k in ["Inflación", "PIB", "IGAE", "Desempleo", "Cetes"]):
        return f"{float(v):,.2f}"
    return f"{float(v):,.2f}"

if not df_vis.empty:
    df_vis["Valor"] = df_vis.apply(fmt_val, axis=1)

st.dataframe(df_vis, use_container_width=True, hide_index=True)

# =========================
# HISTÓRICO 24 MESES (TABLA) — EN VEZ DE INEGI: OECD/FRED
# =========================
st.divider()
st.subheader("🗓️ Histórico mensual (últimos 24 meses)")

start_24m, end_24m = two_full_years_window()
st.caption(f"Ventana automática: {start_24m.strftime('%Y-%m')} → {end_24m.strftime('%Y-%m')}")

frames = []

# Banxico mensual
bx_ids_24m = {
    "Exportaciones": "SE36664",
    "Importaciones": "SE36672",
    "Balanza Petrolera": "SE36683",
    "Cetes 28d": "SF43936",
}
for nom, sid in bx_ids_24m.items():
    try:
        frames.append(banxico_series_24m(sid, nom, start_24m, end_24m))
    except Exception as e:
        st.write(f"Banxico {nom} error: {e}")

# Macro OECD/FRED
try:
    macro24 = build_macro_table_24m()
    macro24_idx = pd.to_datetime(macro24.index + "-01", errors="coerce") + MonthEnd(0)
    macro24_df = macro24.copy()
    macro24_df.insert(0, "Fecha", macro24_idx)
except Exception as e:
    macro24_df = pd.DataFrame()
    st.write(f"Macro (OECD/FRED) error: {e}")

# Convertir macro24 a formato largo
macro_long = pd.DataFrame()
if not macro24_df.empty:
    tmp = macro24_df.copy()
    tmp = tmp.set_index("Fecha")
    tmp = tmp.reset_index().melt(id_vars=["Fecha"], var_name="Indicador", value_name="Valor")
    tmp = tmp.dropna(subset=["Fecha"])
    tmp["Valor"] = pd.to_numeric(tmp["Valor"], errors="coerce")
    tmp = tmp.dropna(subset=["Valor"])
    macro_long = tmp[["Fecha", "Indicador", "Valor"]].copy()

if frames and any(not f.empty for f in frames):
    all_hist = pd.concat([f for f in frames if not f.empty], ignore_index=True)
else:
    all_hist = pd.DataFrame(columns=["Fecha", "Indicador", "Valor"])

if not macro_long.empty:
    all_hist = pd.concat([all_hist, macro_long], ignore_index=True)

if not all_hist.empty:
    all_hist["Fecha"] = pd.to_datetime(all_hist["Fecha"], errors="coerce")
    all_hist["Valor"] = pd.to_numeric(all_hist["Valor"], errors="coerce")
    all_hist = all_hist.dropna(subset=["Fecha", "Valor", "Indicador"])
    all_hist["Fecha"] = all_hist["Fecha"].dt.to_period("M").dt.to_timestamp()

    all_hist = (
        all_hist.sort_values("Fecha")
        .groupby(["Fecha", "Indicador"], as_index=False)["Valor"]
        .last()
    )

    hist = (
        all_hist.pivot_table(index="Fecha", columns="Indicador", values="Valor", aggfunc="last")
        .sort_index()
    )

    idx = pd.period_range(start_24m.to_period("M"), end_24m.to_period("M"), freq="M").to_timestamp()
    hist = hist.reindex(idx)

    # --- CAMBIO A FORMATO HORIZONTAL AQUÍ ---
    # Formateamos la fecha para que se vea limpia como encabezado (YYYY-MM)
    hist.index = hist.index.strftime('%Y-%m')
    
    # Transponemos el DataFrame
    hist_horizontal = hist.transpose()

    # Mostramos la tabla (quitamos hide_index porque ahora los nombres de indicadores están en el index)
    st.dataframe(hist_horizontal, use_container_width=True)

    out2 = io.BytesIO()
    with pd.ExcelWriter(out2, engine="openpyxl") as writer:
        # Guardamos también en horizontal para el Excel
        hist_horizontal.to_excel(writer, index=True, sheet_name="Historico_24m")

    st.download_button(
        label="📥 Descargar Histórico 24 meses (Excel)",
        data=out2.getvalue(),
        file_name=f"Historico_24m_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.warning("Histórico vacío: no se pudieron traer series en la ventana de 24 meses.")
# =========================
# PORTAFOLIO MÍNIMA VARIANZA
# =========================
st.divider()
st.subheader("🧮 Portafolio de Mínima Varianza (S&P 500)")

with st.expander("Configurar y calcular", expanded=True):
    default_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM"]


    SP500_ALL = [str(x).strip().upper() for x in SP500_ALL if pd.notna(x)]
    SP500_ALL = list(dict.fromkeys(SP500_ALL))

    # normaliza defaults y filtra a los que sí existan
    default_tickers = [t.strip().upper() for t in default_tickers]
    default_tickers = [t for t in default_tickers if t in SP500_ALL]
    if len(default_tickers) == 0:
        default_tickers = SP500_ALL[:6]

    seleccion = st.multiselect(
        "Selecciona 2+ tickers",
        options=SP500_ALL,
        default=default_tickers
)

    run_opt = st.button("Calcular mínima varianza")

if run_opt:
    if len(seleccion) < 2:
        st.warning("Selecciona al menos 2 tickers.")
    else:
        with st.spinner("Optimizando (3 años de precios)..."):
            weights, perf = get_min_volatility_portfolio(seleccion)

        if weights is None or perf is None:
            st.warning("No se pudo calcular la optimización.")
        else:
            exp_ret, vol, sharpe = perf
            k1, k2, k3 = st.columns(3)
            k1.metric("Retorno esperado anual", f"{exp_ret*100:,.2f}%")
            k2.metric("Volatilidad anual", f"{vol*100:,.2f}%")
            k3.metric("Sharpe", f"{sharpe:,.3f}")

            wdf = (
                pd.DataFrame(list(weights.items()), columns=["Ticker", "Peso"])
                .query("Peso > 0")
                .sort_values("Peso", ascending=False)
                .reset_index(drop=True)
            )
            wdf["Peso (%)"] = (wdf["Peso"] * 100).round(2)
            wdf = wdf.drop(columns=["Peso"])
            st.dataframe(wdf, use_container_width=True, hide_index=True)

# =========================
# MEJOR PORTAFOLIO (MAX SHARPE BUSCANDO COMBINACIONES)
# =========================
st.divider()
st.subheader("🏆 Mejor Portafolio (Max Sharpe) buscando combinaciones")

with st.expander("Configurar búsqueda", expanded=False):
    k = st.number_input("Número de acciones en el portafolio (k)", min_value=2, max_value=25, value=5, step=1)
    m = st.number_input("Universo preseleccionado (m)", min_value=20, max_value=200, value=80, step=10)
    top_sh = st.number_input("Top por Sharpe individual para candidatos", min_value=50, max_value=500, value=200, step=25)
    trials = st.number_input("Iteraciones (trials)", min_value=1000, max_value=200000, value=20000, step=1000)
    rf = st.number_input("Tasa libre de riesgo (rf)", min_value=0.0, max_value=0.20, value=0.00, step=0.005, format="%.3f")
    seed = st.number_input("Seed", min_value=1, max_value=999999, value=123, step=1)

run_best = st.button("Buscar mejor portafolio (Max Sharpe)")

if run_best:
    with st.spinner("Descargando precios + preselección + búsqueda..."):
        best = option3_best_sp500_combo(
            SP500_ALL,
            k=int(k),
            m=int(m),
            top_sharpe=int(top_sh),
            trials=int(trials),
            rf=float(rf),
            seed=int(seed),
            period="3y",
            min_obs=252*2,
        )

    if best.get("combo") is None:
        st.warning("No se encontró portafolio (datos insuficientes o falló la descarga).")
    else:
        ret, vol = best["perf"]
        st.metric("Sharpe", f"{best['sharpe']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("Retorno esperado anual", f"{ret*100:.2f}%")
        c2.metric("Volatilidad anual", f"{vol*100:.2f}%")

        st.write("**Tickers (combo ganador):**", list(best["combo"]))

        w = best.get("weights_series", pd.Series(dtype=float))
        if not w.empty:
            wdf = w.reset_index()
            wdf.columns = ["Ticker", "Peso"]
            wdf["Peso (%)"] = (wdf["Peso"] * 100).round(2)
            st.dataframe(wdf, use_container_width=True, hide_index=True)
        else:
            st.warning("Se encontró combo, pero los pesos quedaron vacíos.")

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            wdf.to_excel(writer, index=False, sheet_name="Mejor_Portafolio")
        st.download_button(
            "📥 Descargar pesos (Excel)",
            data=out.getvalue(),
            file_name=f"Mejor_Portafolio_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# =========================
# SUSCRIPCIONES
# =========================
st.sidebar.header("📬 Suscripciones")
email_sub = st.sidebar.text_input("Tu correo electrónico:")
if st.sidebar.button("Activar Suscripción"):
    if "@" in email_sub:
        with open("suscriptores.txt", "a", encoding="utf-8") as f:
            f.write(email_sub + "\n")
        st.sidebar.success("Suscrito.")
    else:
        st.sidebar.error("Correo inválido.")
