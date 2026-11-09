SOLAR AIR HEATER — AI DIGITAL TWIN DASHBOARD
================================================

This is a hardware-free software prototype. It supports:
- Synthetic demo simulation (clearly labelled, not measured)
- User CSV upload
- Custom scenario simulation
- Random Forest outlet-temperature prediction
- Isolation Forest anomaly screening
- Explainable AI feature importance
- Conceptual digital-twin heat-flow view
- Day/time fields and CSV downloads
- Responsive Streamlit layout for laptop and phone

RUN ON WINDOWS
1. Extract this ZIP into a normal folder, for example:
   C:\Solar_Air_Heater_AI_Digital_Twin
2. Open VS Code.
3. File > Open Folder > select the extracted folder itself.
4. If VS Code asks "Do you trust the authors?", choose Trust only if this is your own downloaded project.
5. Open Terminal > New Terminal.
6. Run:
   py -m pip install -r requirements.txt
7. Run:
   py -m streamlit run app.py
8. Laptop browser:
   http://localhost:8501

PHONE ON SAME WI-FI
1. Keep the laptop app running.
2. In Command Prompt run: ipconfig
3. Find the IPv4 address under the Wi-Fi adapter.
4. On phone Chrome open: http://YOUR-LAPTOP-IP:8501
   Both devices must be on the same Wi-Fi. Some VPN/firewall/network policies block this.

PUBLISH ONLINE
To get a public streamlit.app address, deploy this folder to Streamlit Community Cloud.
Deployment is not included or already completed by this ZIP.

CSV INPUT COLUMNS
Required:
Inlet Temp (°C), Storage Temp (°C), Humidity (%),
Solar Irradiance (W/m²), Air Velocity (m/s), Outlet Temp (°C)
Optional: Reading, Day, Time

RESEARCH INTEGRITY
Demo data are synthetic and must not be presented as actual experimental measurements.
Model scores from synthetic data are illustrative only. Real validation requires measured data.
