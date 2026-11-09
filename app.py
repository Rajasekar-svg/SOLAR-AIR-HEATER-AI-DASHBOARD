import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Solar Air Heater | AI Digital Twin", page_icon="☀️", layout="wide")

# ---------- Demo physics-inspired dataset (SIMULATED, not experimental) ----------
def make_demo(n=300, seed=42):
    rng = np.random.default_rng(seed)
    minutes = np.arange(n) * 5
    tod = 8 + minutes / 60
    solar = np.clip(650*np.sin(np.pi*(tod-6)/12) + rng.normal(0, 35, n), 0, 950)
    inlet = 24 + 3*np.sin(np.linspace(0, 3*np.pi, n)) + rng.normal(0, 1.1, n)
    humidity = np.clip(58 - 9*np.sin(np.linspace(0, 2*np.pi, n)) + rng.normal(0, 3, n), 25, 90)
    velocity = np.clip(1.25 + 0.18*np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, .12, n), .5, 2.2)
    storage = np.zeros(n)
    storage[0] = 34
    for i in range(1, n):
        storage[i] = np.clip(storage[i-1] + solar[i]*0.0017 - (storage[i-1]-inlet[i])*0.018 + rng.normal(0,.25), 25, 85)
    outlet = inlet + 0.012*solar + 0.16*(storage-inlet) - 2.0*velocity + rng.normal(0, 1.0, n)
    outlet = np.clip(outlet, inlet, 95)
    day = np.array([f"Day {i//100+1}" for i in range(n)])
    time_of_day = [f"{int((8+(i%100)*5/60)):02d}:{int(((i%100)*5)%60):02d}" for i in range(n)]
    return pd.DataFrame({
        "Reading": np.arange(1,n+1), "Day":day, "Time":time_of_day,
        "Inlet Temp (°C)":inlet.round(2), "Storage Temp (°C)":storage.round(2),
        "Humidity (%)":humidity.round(2), "Solar Irradiance (W/m²)":solar.round(2),
        "Air Velocity (m/s)":velocity.round(2), "Outlet Temp (°C)":outlet.round(2)
    })

FEATURES = ["Inlet Temp (°C)", "Storage Temp (°C)", "Humidity (%)", "Solar Irradiance (W/m²)", "Air Velocity (m/s)"]
TARGET = "Outlet Temp (°C)"

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#f5f7ed 0%,#eef6ef 55%,#e8f2e9 100%);}
.hero {background:linear-gradient(120deg,#103f2b,#187344); color:white; padding:28px 30px; border-radius:24px; border-bottom:7px solid #a6e33d; margin-bottom:18px;}
.hero h1 {font-size:2.25rem; margin:0 0 8px 0;}
.hero p {font-size:1rem; letter-spacing:2px; margin:0; color:#d7f6b1;}
.smallnote {font-size:.85rem;color:#637166;}
div[data-testid="stMetric"] {background:white; border:1px solid #d6e5d7; padding:14px; border-radius:16px;}
</style>
<div class="hero">
<h1>☀️ Solar Air Heater — AI Digital Twin</h1>
<p>SIMULATE • PREDICT • ANALYSE • OPTIMISE</p>
</div>
""", unsafe_allow_html=True)

st.caption("Software-only research prototype. No physical sensors or hardware are connected.")
with st.sidebar:
    st.header("⚙️ Experiment controls")
    mode = st.radio("Data mode", ["Demo simulation", "Upload your CSV", "Custom scenario"])
    st.divider()
    st.subheader("Simulation settings")
    daypart = st.select_slider("Sunlight condition", options=["Early morning","Morning","Noon","Afternoon","Evening"], value="Noon")
    fan = st.slider("Air velocity (m/s)", 0.5, 2.5, 1.25, 0.05)
    ambient = st.slider("Inlet temperature (°C)", 10.0, 40.0, 25.0, 0.5)
    humidity_in = st.slider("Humidity (%)", 10, 95, 55)
    storage_in = st.slider("Stored pebble temperature (°C)", 20.0, 90.0, 48.0, 0.5)

if mode == "Upload your CSV":
    file = st.file_uploader("Upload CSV with these columns: " + ", ".join(FEATURES+[TARGET]), type=["csv"])
    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES+[TARGET] if c not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()
        df = df.copy()
        if "Reading" not in df: df.insert(0,"Reading",range(1,len(df)+1))
        if "Day" not in df: df["Day"] = "Uploaded"
        if "Time" not in df: df["Time"] = "Not provided"
        data_label = "USER-UPLOADED DATA"
    else:
        st.info("CSV upload select cheyyakapothe demo data use avuthundi.")
        df = make_demo()
        data_label = "SYNTHETIC DEMO — NOT MEASURED"
elif mode == "Custom scenario":
    solar_map = {"Early morning":180,"Morning":430,"Noon":850,"Afternoon":620,"Evening":120}
    solar_value = solar_map[daypart]
    out_est = ambient + .012*solar_value + .16*(storage_in-ambient) - 2*fan
    df = make_demo()
    df.loc[df.index[-1], FEATURES] = [ambient,storage_in,humidity_in,solar_value,fan]
    df.loc[df.index[-1], TARGET] = max(ambient, out_est)
    df.loc[df.index[-1], ["Day","Time"]] = ["Custom scenario",daypart]
    data_label = "CUSTOM SIMULATED SCENARIO"
else:
    df = make_demo()
    data_label = "SYNTHETIC DEMO — NOT MEASURED"

# Train/test model
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
if len(df) < 12:
    st.error("At least 12 valid rows are needed for model training.")
    st.stop()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)
model = RandomForestRegressor(n_estimators=180, max_depth=8, random_state=42)
model.fit(X_train,y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test,pred)
rmse = mean_squared_error(y_test,pred)**0.5
r2 = r2_score(y_test,pred) if len(y_test)>1 else 0
all_pred = model.predict(X)
df = df.copy()
df["RF Predicted Outlet Temp (°C)"] = all_pred.round(2)
residual = np.abs(y-all_pred)

# Anomaly model on sensor feature values
iso = IsolationForest(contamination=0.04, random_state=42)
df["Anomaly Status"] = np.where(iso.fit_predict(X)==-1, "Anomaly", "Normal")
df["Prediction Error (°C)"] = residual.round(2)

st.markdown(f"**DATA STATUS:** `{data_label}`")
if "SYNTHETIC" in data_label:
    st.warning("Demo values are simulated for software demonstration. They are NOT real experimental measurements. Replace with actual sensor readings before reporting experimental results.")

latest = df.iloc[-1]
k1,k2,k3,k4 = st.columns(4)
k1.metric("Outlet temperature",f"{latest[TARGET]:.2f} °C")
k2.metric("AI predicted outlet",f"{latest['RF Predicted Outlet Temp (°C)']:.2f} °C")
k3.metric("Thermal storage",f"{latest['Storage Temp (°C)']:.2f} °C")
k4.metric("Anomaly readings",f"{(df['Anomaly Status']=='Anomaly').sum()}")

tabs = st.tabs(["🏠 Dashboard","🧊 Digital Twin","🤖 AI Lab","📈 Research Analytics","🗂️ Dataset","ℹ️ Methodology"])
with tabs[0]:
    st.subheader("Live-style system overview")
    c1,c2 = st.columns(2)
    with c1:
        fig=px.line(df,x="Reading",y=["Inlet Temp (°C)","Outlet Temp (°C)","Storage Temp (°C)"],title="Temperature trends")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.line(df,x="Reading",y=["Solar Irradiance (W/m²)"],title="Solar irradiance trend")
        st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        fig=px.line(df,x="Reading",y="Air Velocity (m/s)",title="Air velocity trend")
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        fig=px.line(df,x="Reading",y="Humidity (%)",title="Humidity trend")
        st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.subheader("Conceptual Digital Twin — simulated heat flow")
    st.caption("Animation/values are conceptual simulations, not a live camera or measured physical model.")
    solar_now=float(latest["Solar Irradiance (W/m²)"])
    storage_now=float(latest["Storage Temp (°C)"])
    inlet_now=float(latest["Inlet Temp (°C)"])
    outlet_now=float(latest[TARGET])
    fig=go.Figure()
    fig.add_shape(type="rect",x0=0.08,y0=.25,x1=.92,y1=.78,line=dict(color="#245b3d",width=3),fillcolor="#d8e8c7")
    fig.add_shape(type="rect",x0=.14,y0=.32,x1=.86,y1=.42,line=dict(color="#333"),fillcolor="#343434")
    fig.add_shape(type="rect",x0=.14,y0=.45,x1=.86,y1=.65,line=dict(color="#9b6b35"),fillcolor="#c9a16a")
    for i in range(12):
        x=.18+i*.056
        fig.add_shape(type="circle",x0=x,y0=.49+(i%3)*.045,x1=x+.025,y1=.515+(i%3)*.045,line=dict(color="#87602e"),fillcolor="#98703d")
    fig.add_annotation(x=.5,y=.86,text="☀️ Solar energy input",showarrow=False,font=dict(size=16,color="#b36b00"))
    fig.add_annotation(x=.5,y=.70,text=f"Absorber + pebble storage  |  {storage_now:.1f} °C",showarrow=False,font=dict(size=13,color="#3a2b1c"))
    fig.add_annotation(x=.5,y=.18,text=f"Airflow: {inlet_now:.1f} °C  →  {outlet_now:.1f} °C",showarrow=False,font=dict(size=15,color="#17623c"))
    fig.update_xaxes(visible=False,range=[0,1]); fig.update_yaxes(visible=False,range=[0,1]); fig.update_layout(height=390,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.write(f"Solar input: **{solar_now:.1f} W/m²** · Inlet: **{inlet_now:.1f} °C** · Outlet: **{outlet_now:.1f} °C**")
    st.info("Thermal battery estimate is represented by storage temperature. Actual stored energy (kWh) needs measured mass, material heat capacity and temperature difference.")

with tabs[2]:
    st.subheader("AI prediction and anomaly detection")
    a,b,c=st.columns(3); a.metric("MAE",f"{mae:.2f} °C"); b.metric("RMSE",f"{rmse:.2f} °C"); c.metric("R²",f"{r2:.3f}")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index,y=y_test,name="Actual",mode="lines+markers"))
    fig.add_trace(go.Scatter(x=y_test.index,y=pred,name="RF prediction",mode="lines+markers"))
    fig.update_layout(title="Random Forest: actual vs predicted (held-out test rows)",xaxis_title="Test row index",yaxis_title="Outlet temperature (°C)")
    st.plotly_chart(fig,use_container_width=True)
    imp=pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Explainable AI: feature importance"),use_container_width=True)
    st.caption("Feature importance shows model reliance, not proof of physical causation.")
    st.subheader("Anomaly detection")
    st.plotly_chart(px.scatter(df,x="Reading",y=TARGET,color="Anomaly Status",title="Isolation Forest: unusual outlet-temperature patterns"),use_container_width=True)
    st.dataframe(df[df["Anomaly Status"]=="Anomaly"][["Reading","Day","Time"]+FEATURES+[TARGET]],use_container_width=True)

with tabs[3]:
    st.subheader("Research analytics")
    delta=(df[TARGET]-df["Inlet Temp (°C)"]).mean()
    st.metric("Average temperature rise",f"{delta:.2f} °C")
    st.metric("Maximum outlet temperature",f"{df[TARGET].max():.2f} °C")
    st.metric("Average solar irradiance",f"{df['Solar Irradiance (W/m²)'].mean():.1f} W/m²")
    st.plotly_chart(px.scatter(df,x="Solar Irradiance (W/m²)",y=TARGET,color="Storage Temp (°C)",title="Outlet temperature vs solar irradiance"),use_container_width=True)
    st.plotly_chart(px.box(df,x="Day",y=TARGET,title="Outlet temperature distribution by day"),use_container_width=True)
    st.caption("Thermal efficiency is not calculated because useful heat output requires air mass-flow rate and a validated measurement method.")

with tabs[4]:
    st.subheader("Experiment dataset")
    st.dataframe(df,use_container_width=True,height=420)
    st.download_button("⬇️ Download analysed CSV",df.to_csv(index=False).encode("utf-8"),"solar_air_heater_analysis.csv","text/csv")
    st.download_button("⬇️ Download sample input template",pd.DataFrame(columns=["Reading","Day","Time"]+FEATURES+[TARGET]).to_csv(index=False).encode("utf-8"),"solar_air_heater_csv_template.csv","text/csv")

with tabs[5]:
    st.subheader("Project methodology")
    st.markdown("""
    **System concept:** Solar radiation heats the absorber plate. Sensible heat storage (pebbles/rocks) stores part of the thermal energy. Air passing through the collector gains heat.

    **Software pipeline:** Simulated or uploaded CSV → data validation → Random Forest regression → outlet-temperature prediction → Isolation Forest anomaly screening → graphs and downloadable results.

    **Required CSV columns:** `Inlet Temp (°C)`, `Storage Temp (°C)`, `Humidity (%)`, `Solar Irradiance (W/m²)`, `Air Velocity (m/s)`, `Outlet Temp (°C)`. Optional columns: `Reading`, `Day`, `Time`.

    **Limitations:** This version is hardware-free. Demo data are synthetic. Model scores on synthetic data do not establish real-world accuracy. Validate with real experimental readings before drawing research conclusions.
    """)
