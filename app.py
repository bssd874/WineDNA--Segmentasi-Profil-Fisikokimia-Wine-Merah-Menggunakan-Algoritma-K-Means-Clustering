from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="WineDNA Clustering",
    page_icon="🍷",
    layout="wide"
)


@st.cache_resource
def load_model():
    lokasi_model = (
        Path(__file__).parent
        / "deployment_wine"
        / "model_kmeans_wine.joblib"
    )

    return joblib.load(lokasi_model)


def prediksi_cluster(data_baru, paket_model):
    fitur = paket_model["fitur_clustering"]
    fitur_transformasi = paket_model["fitur_transformasi"]

    X_baru = data_baru[fitur].copy()

    if X_baru.isnull().any().any():
        raise ValueError("Data mengandung missing value.")

    if not np.isfinite(X_baru.to_numpy()).all():
        raise ValueError("Data mengandung nilai tidak valid.")

    # Transformasi Yeo-Johnson
    X_transformed = X_baru.copy()

    X_transformed[fitur_transformasi] = (
        paket_model["power_transformer"]
        .transform(X_transformed[fitur_transformasi])
    )

    # Standardisasi
    X_scaled = pd.DataFrame(
        paket_model["scaler"].transform(X_transformed),
        columns=fitur,
        index=X_baru.index
    )

    # Prediksi cluster
    cluster = int(
        paket_model["model"]
        .predict(X_scaled)[0]
    )

    # Menghitung jarak menuju centroid
    jarak = (
        paket_model["model"]
        .transform(X_scaled)[0]
    )

    jarak_terdekat = float(jarak.min())

    return cluster, jarak_terdekat


paket_model = load_model()

st.title("🍷 WineDNA")
st.subheader("Segmentasi Profil Fisikokimia Wine Merah")

st.write(
    """
    Aplikasi ini mengelompokkan sampel wine merah berdasarkan
    11 karakteristik fisikokimia menggunakan algoritma K-Means.
    """
)

st.info(
    "Aplikasi ini melakukan clustering, bukan memprediksi nilai quality."
)

with st.form("form_wine"):

    kolom_1, kolom_2 = st.columns(2)

    with kolom_1:
        fixed_acidity = st.number_input(
            "Fixed Acidity",
            min_value=0.0,
            value=8.30,
            step=0.10
        )

        volatile_acidity = st.number_input(
            "Volatile Acidity",
            min_value=0.0,
            value=0.53,
            step=0.01
        )

        citric_acid = st.number_input(
            "Citric Acid",
            min_value=0.0,
            value=0.27,
            step=0.01
        )

        residual_sugar = st.number_input(
            "Residual Sugar",
            min_value=0.0,
            value=2.50,
            step=0.10
        )

        chlorides = st.number_input(
            "Chlorides",
            min_value=0.0,
            value=0.087,
            step=0.001,
            format="%.3f"
        )

        free_sulfur_dioxide = st.number_input(
            "Free Sulfur Dioxide",
            min_value=0.0,
            value=15.0,
            step=1.0
        )

    with kolom_2:
        total_sulfur_dioxide = st.number_input(
            "Total Sulfur Dioxide",
            min_value=0.0,
            value=46.0,
            step=1.0
        )

        density = st.number_input(
            "Density",
            min_value=0.0,
            value=0.9967,
            step=0.0001,
            format="%.4f"
        )

        ph = st.number_input(
            "pH",
            min_value=0.0,
            value=3.31,
            step=0.01
        )

        sulphates = st.number_input(
            "Sulphates",
            min_value=0.0,
            value=0.66,
            step=0.01
        )

        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            value=10.40,
            step=0.10
        )

    tombol_prediksi = st.form_submit_button(
        "Analisis Cluster",
        use_container_width=True
    )


if tombol_prediksi:

    sampel = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    try:
        cluster, jarak = prediksi_cluster(
            sampel,
            paket_model
        )

        nama = paket_model["nama_cluster"][cluster]

        st.success(f"Sampel termasuk Cluster {cluster}")
        st.subheader(nama)

        metrik_1, metrik_2 = st.columns(2)

        metrik_1.metric(
            "Cluster",
            cluster
        )

        metrik_2.metric(
            "Jarak ke Centroid",
            f"{jarak:.4f}"
        )

        st.write("#### Data sampel")
        st.dataframe(
            sampel,
            use_container_width=True
        )

        if "profil_cluster" in paket_model:
            st.write("#### Profil rata-rata cluster")

            profil = (
                paket_model["profil_cluster"]
                .loc[cluster]
                .rename("Nilai rata-rata")
                .to_frame()
            )

            st.dataframe(
                profil,
                use_container_width=True
            )

        st.caption(
            """
            Jarak centroid menunjukkan kedekatan sampel dengan pusat
            cluster dan bukan nilai confidence atau akurasi.
            """
        )

    except Exception as error:
        st.error(f"Prediksi gagal: {error}")