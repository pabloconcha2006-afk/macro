#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 02:36:18 2026

@author: pabloconcha
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import io
import os
import urllib3
from datetime import datetime
import yfinance as yf

# Se usan en get_min_volatility_portfolio
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# Desactiva los errores de conexión segura para que INEGI no nos bloquee
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# CONFIGURACIÓN (RUTAS Y TOKENS)
# ==========================================
SP500_ALL = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMBN", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AYI", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CW", "D", "DAL", "DAN", "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "ERTS", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEV", "GGE", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWRE", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTU", "INVH", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDX", "NEE", "NEM", "NFG", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTR", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "RWD", "SBAC", "SBUX", "SCHW", "SHW", "SIRI", "SITM", "SJM", "SKW", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFT", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VMC", "VTRS", "VRTX", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
]
STPS_XLS_PATH = "/Users/pabloconcha/Library/spyder-6/envs/spyder-runtime/macro_html/stps_xls/302_0074.xls"
BANXICO_TOKEN = "710446e886e30952133c0d23a4882d24fb2aafaa659b416b07a05d7b19d2a10f"
INEGI_TOKEN = "852ed086-a989-499d-ad0b-59640f500062"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def get_min_volatility_portfolio(tickers):
    if len(tickers) < 2:
        return None, None

    try:
        data = yf.download(tickers, period="3y", progress=False)

        # Adj Close si existe; si no, Close
        if "Adj Close" in data.columns:
            precios = data["Adj Close"]
        else:
            precios = data["Close"]

        # Limpieza básica
        if isinstance(precios, pd.DataFrame):
            precios = precios.dropna(how="all")

        mu = expected_returns.mean_historical_return(precios)
        S = risk_models.sample_cov(precios)

        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance()  # (ret, vol, sharpe)

        return cleaned_weights, perf
    except Exception as e:
        st.error(f"Error interno en la optimización: {e}")
        return None, None

def fetch_all_data():
    results = []

    # --- 1. SALARIO MÍNIMO (STPS LOCAL) ---
    try:
        df_stps = pd.read_excel(STPS_XLS_PATH, sheet_name="Salario Mínimo 2019-2025", engine="xlrd")
        val_salario = pd.to_numeric(df_stps.stack(), errors="coerce").dropna().iloc[-1]
        results.append({
            "Fuente": "STPS (Local)", "Indicador": "Salario Mínimo",
            "Valor": float(val_salario), "Fecha": datetime.today().strftime("%Y-%m-%d")
        })
    except Exception as e:
        results.append({"Fuente": "STPS", "Indicador": "Salario Mínimo", "Valor": 0.0, "Fecha": f"Error: {str(e)}"})

    # --- 2. BANXICO (API) ---
    bx_ids = {
        "Exportaciones": "SE36664",
        "Importaciones": "SE36672",
        "Balanza Petrolera": "SE36683",
        "Cetes 28d": "SF43936"
    }
    for nom, sid in bx_ids.items():
        try:
            r = requests.get(
                f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{sid}/datos/oportuno",
                headers={"Bmx-Token": BANXICO_TOKEN, "User-Agent": UA},
                timeout=15
            )
            d = r.json()["bmx"]["series"][0]["datos"][0]
            results.append({
                "Fuente": "Banxico", "Indicador": nom,
                "Valor": float(str(d["dato"]).replace(",", "")), "Fecha": d["fecha"]
            })
        except:
            pass

    # --- 3. INEGI (API) ---
    inegi_config = {
        "6200011881": ("Inflación General", "BISE"),
        "6207132027": ("PIB Anual", "BISE"),
        "6200093972": ("Tasa Desempleo", "BISE"),
        "494056": ("IGAE (Actividad Econ)", "BIE"),
        "493666": ("Actividad Industrial", "BIE"),
        "493668": ("Ind. Manufacturera", "BIE")
    }

    for sid, (nom, sistema) in inegi_config.items():
        try:
            url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/json/INDICATOR/{sid}/es/0700/false/{sistema}/2.0/{INEGI_TOKEN}?type=json"
            response = requests.get(url, headers={"User-Agent": UA}, timeout=20, verify=False)
            if response.status_code == 200:
                data = response.json()
                obs = data["Series"][0]["OBSERVATIONS"][-1]
                results.append({
                    "Fuente": f"INEGI ({sistema})",
                    "Indicador": nom,
                    "Valor": float(obs["OBS_VALUE"]),
                    "Fecha": obs["TIME_PERIOD"]
                })
        except:
            results.append({"Fuente": "INEGI", "Indicador": nom, "Valor": 0.0, "Fecha": "Error de Conexión"})

    return pd.DataFrame(results)

# ==========================================
# INTERFAZ
# ==========================================
st.set_page_config(page_title="Macro Mexico Pro", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0e1117; border: 1px solid #262730; padding: 15px; border-radius: 10px; }
    [data-testid="stMetricValue"] { color: #00d4ff; font-size: 32px; font-weight: bold; }
    .main { background-color: #050505; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Monitor Económico: México 2026")
st.caption("Datos integrados de STPS (Excel Local), Banxico e INEGI (API Real Time)")

with st.spinner("Sincronizando con fuentes gubernamentales..."):
    df = fetch_all_data()

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)

    def get_val(nombre):
        temp = df[df["Indicador"].str.contains(nombre, case=False, na=False)]
        return temp.iloc[0]["Valor"] if not temp.empty else 0.0

    c1.metric("💰 Salario Mínimo", f"${get_val('Salario'):,.2f}")
    c2.metric("🏷️ Inflación", f"{get_val('Inflación'):,.2f}%")
    c3.metric("📉 Cetes 28d", f"{get_val('Cetes'):,.2f}%")
    c4.metric("🏗️ PIB / IGAE", f"{get_val('IGAE'):,.2f}%")

    st.divider()

    st.subheader("📋 Detalle de Indicadores")
    df_vis = df.copy()
    df_vis["Valor"] = df_vis.apply(
        lambda x: f"{x['Valor']:,.2f}%"
        if any(p in x["Indicador"] for p in ["%", "Inflación", "PIB", "IGAE", "Desempleo", "Cetes"])
        else f"${x['Valor']:,.2f}",
        axis=1
    )
    st.dataframe(df_vis, use_container_width=True, hide_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Reporte_Macro")

    st.download_button(
        label="📥 Descargar Reporte Completo (Excel)",
        data=output.getvalue(),
        file_name=f"Macro_Mexico_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==========================================
# NUEVO: SECCIÓN MÍNIMA VARIANZA (APARECE EN LA WEB)
# ==========================================
st.divider()
st.subheader("🧮 Portafolio de Mínima Varianza (S&P 500)")

with st.expander("Configurar y calcular", expanded=True):
    # Default conservador (6 tickers) para que cargue rápido y no falle por límites de Yahoo
    default_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM"]
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
            # perf = (expected_return, volatility, sharpe)
            exp_ret, vol, sharpe = perf

            k1, k2, k3 = st.columns(3)
            k1.metric("Retorno esperado anual", f"{exp_ret*100:,.2f}%")
            k2.metric("Volatilidad anual", f"{vol*100:,.2f}%")
            k3.metric("Sharpe", f"{sharpe:,.3f}")

            st.markdown("**Pesos (cleaned)**")
            wdf = (
                pd.DataFrame(list(weights.items()), columns=["Ticker", "Peso"])
                .query("Peso > 0")
                .sort_values("Peso", ascending=False)
                .reset_index(drop=True)
            )


            wdf["Peso (%)"] = wdf["Peso"] * 100
            wdf["Peso (%)"] = wdf["Peso (%)"].round(2)
            wdf = wdf.drop(columns=["Peso"])
            st.dataframe(wdf, use_container_width=True, hide_index=True)

# --- SECCIÓN DE SUSCRIPCIÓN ---
st.sidebar.header("📬 Suscripciones")
st.sidebar.info("Recibe este reporte automáticamente cada mes y cuando los datos cambien.")
email_sub = st.sidebar.text_input("Tu correo electrónico:")
if st.sidebar.button("Activar Suscripción"):
    if "@" in email_sub:
        with open("suscriptores.txt", "a") as f:
            f.write(email_sub + "\n")
        st.sidebar.success("¡Suscrito con éxito!")
    else:
        st.sidebar.error("Ingresa un correo válido.")
