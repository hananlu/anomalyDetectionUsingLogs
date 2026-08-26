import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------------------------------------
# 1. Konfigurasi Tampilan Halaman Streamlit
# -------------------------------------------------------------
st.set_page_config(
    page_title="Network Anomaly & Threat Detector",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------------------
# 2. Fungsi Load Model Pipeline (Di-cache agar performa cepat)
# -------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load('anomaly_detection_pipeline.joblib')

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(
        f"Gagal memuat file model `anomaly_detection_pipeline.joblib`. "
        f"Pastikan file berada di satu folder yang sama dengan app.py. Error: {e}"
    )
    st.stop()

# -------------------------------------------------------------
# 3. Header & Antarmuka Utama
# -------------------------------------------------------------
st.title("🛡️ Network Anomaly & Threat Detection System")
st.markdown(
    "Aplikasi inferensi Machine Learning berbasis **Random Forest** untuk mendeteksi "
    "aktivitas anomali dan ancaman pada lalu lintas log jaringan."
)

st.sidebar.header("⚙️ Input Parameter Log Jaringan")

# -------------------------------------------------------------
# 4. Form Input Data Log di Sidebar
# -------------------------------------------------------------
with st.sidebar:
    st.subheader("1. Waktu Kejadian")
    input_date = st.date_input("Tanggal Log", value=datetime.today())
    input_time = st.time_input("Waktu Log (Jam:Menit)", value=datetime.now().time())

    st.subheader("2. Alamat IP")
    source_ip = st.text_input("Source IP", value="192.168.1.124")
    dest_ip = st.text_input("Destination IP", value="192.168.1.167")

    st.subheader("3. Metadata Koneksi")
    protocol = st.selectbox("Protocol", ["HTTP", "HTTPS", "TCP", "UDP", "ICMP"])
    action = st.selectbox("Action", ["allowed", "blocked"])
    log_type = st.selectbox("Log Type", ["firewall", "ids", "application"])
    bytes_transferred = st.number_input("Bytes Transferred", min_value=0, value=15757, step=500)

    st.subheader("4. Payload / Client Information")
    user_agent = st.selectbox(
        "User Agent",
        [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/535.11",
            "curl/7.64.1",
            "Nmap Scripting Engine",
            "SQLMap/1.6-dev"
        ]
    )
    request_path = st.text_input("Request Path", value="/api/login")

    st.subheader("5. Pengaturan Model")
    threshold = st.slider(
        "Threshold Klasifikasi Threat",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Turunkan threshold (misal 0.35) untuk menaikkan sensitivitas deteksi (Recall tinggi)."
    )

# -------------------------------------------------------------
# 5. Preprocessing & Feature Engineering Input
# -------------------------------------------------------------
log_dt = datetime.combine(input_date, input_time)
hour = log_dt.hour
dayofweek = log_dt.weekday()

# Subnet extraction (2 oktet pertama)
src_subnet = '.'.join(str(source_ip).strip().split('.')[:2]) if '.' in str(source_ip) else str(source_ip)
dest_subnet = '.'.join(str(dest_ip).strip().split('.')[:2]) if '.' in str(dest_ip) else str(dest_ip)

# Path features extraction
path_str = str(request_path).strip()
path_depth = path_str.count('/')
path_is_root = 1 if path_str == '/' else 0
path_is_login = 1 if any(k in path_str.lower() for k in ['login', 'auth', 'admin']) else 0

# Membentuk DataFrame sesuai skema training
input_df = pd.DataFrame([{
    'protocol': protocol,
    'action': action,
    'log_type': log_type,
    'user_agent': user_agent,
    'src_subnet': src_subnet,
    'dest_subnet': dest_subnet,
    'bytes_transferred': bytes_transferred,
    'hour': hour,
    'dayofweek': dayofweek,
    'path_depth': path_depth,
    'path_is_root': path_is_root,
    'path_is_login': path_is_login
}])

# -------------------------------------------------------------
# 6. Tampilan Hasil Prediksi
# -------------------------------------------------------------
st.subheader("📋 Ringkasan Data Log Masuk")
st.dataframe(input_df, use_container_width=True)

if st.button("🔍 Analisis & Prediksi Keamanan", type="primary", use_container_width=True):
    # Prediksi probabilitas anomali / threat
    proba_threat = pipeline.predict_proba(input_df)[0][1]
    is_threat = proba_threat >= threshold

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Skor Probabilitas Ancaman",
            value=f"{proba_threat * 100:.2f}%",
            delta=f"Threshold: {threshold * 100:.1f}%"
        )
        st.progress(float(proba_threat))

    with col2:
        if is_threat:
            st.error("⚠️ **HASIL: THREAT / ANOMALI TERDETEKSI!**")
            st.markdown(
                f"- **Tindakan:** Segera lakukan isolasi terhadap IP `{source_ip}`.\n"
                f"- **Alasan:** Aktivitas pada endpoint `{request_path}` menunjukkan deviasi perilaku."
            )
        else:
            st.success("✅ **HASIL: LOG NORMAL (BENIGN)**")
            st.markdown("- **Tindakan:** Lalu lintas data diizinkan sesuai kebijakan normal.")

    st.divider()
    with st.expander("Lihat Detail Fitur yang Dikirim ke Model"):
        st.json(input_df.to_dict(orient='records')[0])
