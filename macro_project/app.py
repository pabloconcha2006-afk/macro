#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 02:36:18 2026

@author: pabloconcha
"""

# -*- coding: utf-8 -*-
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import urllib3
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# CONFIG
# =========================
SP500_ALL = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES",
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

STPS_XLS_PATH = "stps_xls/302_0074.xls"  # ruta relativa (repo)

BANXICO_TOKEN = "710446e886e30952133c0d23a4882d24fb2aafaa659b416b07a05d7b19d2a10f"
INEGI_TOKEN = "852ed086-a989-499d-ad0b-59640f500062"
UA = "Mozilla/5.0"

# =========================
# HELPERS 24 MESES
# =========================
def two_full_years_window(today=None):
    today = today or pd.Timestamp.today().normalize()
    end = today.replace(day=1) - pd.Timedelta(days=1)          # fin del mes completo anterior
    start = (end - pd.DateOffset(months=23)).replace(day=1)    # 24 meses contando el mes inicial
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

def inegi_series_24m(indicador_id: str, nombre: str, sistema: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = (
        "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/json/"
        f"INDICATOR/{indicador_id}/es/0700/false/{sistema}/2.0/{INEGI_TOKEN}?type=json"
    )
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    data = r.json()
    obs = data["Series"][0]["OBSERVATIONS"]

    rows = []
    for o in obs:
        tp = str(o.get("TIME_PERIOD", "")).replace("/", "-")
        fecha = pd.to_datetime(tp + "-01", errors="coerce")
        val = pd.to_numeric(o.get("OBS_VALUE"), errors="coerce")
        if pd.notna(fecha) and pd.notna(val):
            rows.append({"Indicador": nombre, "Fecha": fecha, "Valor": float(val)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Fecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("Fecha").groupby(["Fecha", "Indicador"], as_index=False)["Valor"].last()
    df = df[(df["Fecha"] >= start) & (df["Fecha"] <= end)]
    return df

# =========================
# ÚLTIMO DATO (TABLA DETALLE)
# =========================
def fetch_all_data() -> pd.DataFrame:
    results = []

    # STPS local
    try:
        df_stps = pd.read_excel(STPS_XLS_PATH, sheet_name="Salario Mínimo 2019-2025", engine="xlrd")
        val_salario = pd.to_numeric(df_stps.stack(), errors="coerce").dropna().iloc[-1]
        results.append(
            {"Fuente": "STPS", "Indicador": "Salario Mínimo", "Valor": float(val_salario), "Fecha": datetime.today().strftime("%Y-%m-%d")}
        )
    except Exception as e:
        results.append({"Fuente": "STPS", "Indicador": "Salario Mínimo", "Valor": 0.0, "Fecha": f"Error: {e}"})

    # Banxico oportuno
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
            results.append({"Fuente": "Banxico", "Indicador": nom, "Valor": 0.0, "Fecha": f"Error: {e}"})

    # INEGI último
    inegi_last = {
        "6200011881": ("Inflación General", "BISE"),
        "6200093972": ("Tasa Desempleo", "BISE"),
        "494056": ("IGAE (Actividad Econ)", "BIE"),
        "6207132027": ("PIB Anual", "BISE"),
    }
    for sid, (nom, sistema) in inegi_last.items():
        try:
            url = (
                "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/json/"
                f"INDICATOR/{sid}/es/0700/false/{sistema}/2.0/{INEGI_TOKEN}?type=json"
            )
            r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
            r.raise_for_status()
            data = r.json()
            last = data["Series"][0]["OBSERVATIONS"][-1]
            results.append(
                {"Fuente": f"INEGI ({sistema})", "Indicador": nom, "Valor": float(last["OBS_VALUE"]), "Fecha": last["TIME_PERIOD"]}
            )
        except Exception as e:
            results.append({"Fuente": "INEGI", "Indicador": nom, "Valor": 0.0, "Fecha": f"Error: {e}"})

    return pd.DataFrame(results)

# =========================
# MIN VAR
# =========================
def get_min_volatility_portfolio(tickers):
    if len(tickers) < 2:
        return None, None
    try:
        data = yf.download(tickers, period="3y", progress=False)
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
st.caption("STPS (archivo), Banxico e INEGI (API)")

with st.spinner("Sincronizando..."):
    df = fetch_all_data()

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)

    def get_val(nombre):
        temp = df[df["Indicador"].astype(str).str.contains(nombre, case=False, na=False)]
        return float(temp.iloc[0]["Valor"]) if not temp.empty else 0.0

    c1.metric("💰 Salario Mínimo", f"${get_val('Salario'):,.2f}")
    c2.metric("🏷️ Inflación", f"{get_val('Inflación'):,.2f}%")
    c3.metric("📉 Cetes 28d", f"{get_val('Cetes'):,.2f}%")
    c4.metric("🏗️ IGAE", f"{get_val('IGAE'):,.2f}%")

st.divider()
st.subheader("📋 Detalle de Indicadores (último dato)")

df_vis = df.copy()

def fmt_val(row):
    ind = str(row.get("Indicador", ""))
    v = pd.to_numeric(row.get("Valor"), errors="coerce")
    if pd.isna(v):
        return str(row.get("Valor"))
    if any(k in ind for k in ["Inflación", "PIB", "IGAE", "Desempleo", "Cetes"]):
        return f"{float(v):,.2f}%"
    return f"${float(v):,.2f}"

if not df_vis.empty:
    df_vis["Valor"] = df_vis.apply(fmt_val, axis=1)
st.dataframe(df_vis, use_container_width=True, hide_index=True)

# =========================
# HISTÓRICO 24 MESES
# =========================
st.divider()
st.subheader("🗓️ Histórico mensual (últimos 24 meses)")

start_24m, end_24m = two_full_years_window()
st.caption(f"Ventana automática: {start_24m.strftime('%Y-%m')} → {end_24m.strftime('%Y-%m')}")

frames = []

# Banxico 24m
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

# INEGI 24m
inegi_cfg_24m = {
    "6200011881": ("Inflación General", "BISE"),
    "6200093972": ("Tasa Desempleo", "BISE"),
    "494056": ("IGAE", "BIE"),
}
for sid, (nom, sistema) in inegi_cfg_24m.items():
    try:
        frames.append(inegi_series_24m(sid, nom, sistema, start_24m, end_24m))
    except Exception as e:
        st.write(f"INEGI {nom} error: {e}")

if frames and any(not f.empty for f in frames):
    all_hist = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # Normaliza mensual (doble seguridad)
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

    hist_out = hist.reset_index().rename(columns={"index": "Fecha"})
    hist_out["Fecha"] = pd.to_datetime(hist_out["Fecha"]).dt.to_period("M").astype(str)

    st.dataframe(hist_out, use_container_width=True, hide_index=True)

    out2 = io.BytesIO()
    with pd.ExcelWriter(out2, engine="openpyxl") as writer:
        hist_out.to_excel(writer, index=False, sheet_name="Historico_24m")

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
    seleccion = st.multiselect("Selecciona 2+ tickers", options=SP500_ALL, default=default_tickers)
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
