import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6; /* Warna latar belakang */
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        color: #ff6f61;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-title {
        color: #4CAF50;
        font-size: 24px;
        margin-top: 20px;
    }
    .sidebar .block-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Mengatur konfigurasi sidebar
st.sidebar.title("Navigasi")
st.sidebar.markdown("<div style='color: #4CAF50; font-weight: bold;'>🚀 Pilih Halaman:</div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Upload Data", "Analisis", "Prediksi"])


# Membaca data langsung dari file yang diimpor
data = pd.read_csv("Regression.csv")

# Halaman Home
if page == "Home":
    st.markdown("<div class='main-title'>🏠 Selamat Datang di Aplikasi Analisis Data</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 18px; margin-top: 20px;'>Silakan pilih opsi navigasi di sebelah kiri untuk mulai bekerja dengan data Anda.</div>", unsafe_allow_html=True)

    # URL gambar
    image_url = "https://i.imgur.com/lhuz5f4.png"

    # Menampilkan gambar
    st.image(image_url, caption="Welcome to App")


# Halaman Upload Data
elif page == "Upload Data":
    st.markdown("<div class='main-title'>📂 Upload Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Data yang diunggah:</div>", unsafe_allow_html=True)
    st.dataframe(data.head(), height=300)
    st.markdown("<div class='section-title'>Statistik Deskriptif:</div>", unsafe_allow_html=True)
    st.write(data.describe())

# Halaman Analisis
elif page == "Analisis":
    st.markdown("<div class='main-title'>📊 Analisis Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Pilih Fitur dan Target:</div>", unsafe_allow_html=True)

    fitur = st.multiselect("Pilih kolom fitur:", options=data.columns)
    target = st.selectbox("Pilih kolom target:", options=data.columns)

    if fitur and target:
        try:
            X = data[fitur].apply(pd.to_numeric, errors='coerce').fillna(0)
            y = pd.to_numeric(data[target], errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Error dalam memproses data: {e}")
            st.stop()

        if X.isnull().values.any() or y.isnull().values.any():
            st.error("Data mengandung nilai yang tidak valid atau kosong. Harap periksa dataset Anda.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown("<div class='section-title'>Evaluasi Model:</div>", unsafe_allow_html=True)
        st.write(f"**Mean Squared Error (MSE):** {mse}")
        st.write(f"**R-squared (R2):** {r2}")

        st.markdown("<div class='section-title'>Koefisien Model:</div>", unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            'Fitur': fitur,
            'Koefisien': model.coef_
        })
        st.write(coef_df)

# Halaman Prediksi
elif page == "Prediksi":
    st.markdown("<div class='main-title'>🔮 Prediksi Data Baru</div>", unsafe_allow_html=True)
    fitur = st.sidebar.multiselect("Pilih kolom fitur untuk prediksi:", options=data.columns)
    target = st.sidebar.selectbox("Pilih kolom target:", options=data.columns)

    if fitur and target:
        st.markdown("<div class='section-title'>Masukkan Nilai Fitur:</div>", unsafe_allow_html=True)
        input_data = {f: st.number_input(f"Masukkan nilai untuk {f}", value=0.0) for f in fitur}

        if st.button("Prediksi"):
            input_df = pd.DataFrame([input_data])
            model = LinearRegression()
            model.fit(data[fitur].apply(pd.to_numeric, errors='coerce').fillna(0),
                      pd.to_numeric(data[target], errors='coerce').fillna(0))
            prediction = model.predict(input_df)[0]
            st.markdown(f"<div class='section-title'>Prediksi untuk data baru:</div>", unsafe_allow_html=True)
            st.write(f"**{prediction:.2f}**")
