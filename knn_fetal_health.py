import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('janin.sav', 'rb'))

st.title('Klasifikasi Kondisi Janin Menggunakan Metode KNN')

col1, col2 = st.columns(2)
with col1:
    histogram_width = st.number_input('Lebar histogram selama pemeriksaan')
    histogram_max = st.number_input('Nilai histogram tertinggi')
    histogram_number_of_zeroes = st.number_input('Jumlah angka nol dalam pengujian histogram')
    histogram_mean = st.number_input('Nilai rata-rata histrogram')

with col2:
    histogram_min = st.number_input('Nilai histogram terendah')
    histogram_number_of_peaks = st.number_input('Jumlah puncak dalam pengujian histogram')
    histogram_mode = st.number_input('Nilai histrogram sering muncul')
    histogram_median = st.number_input('Nilai tengah histrogram')

prediksi = ''
if st.button('Hasil Prediksi'):
    prediksi = model.predict([[histogram_width, histogram_min, histogram_max, histogram_number_of_peaks,
                               histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median]])

    if (prediksi[0] == 1):
        prediksi = 'kondisi janin dalam keadaan normal'
    elif (prediksi == 2):
        prediksi = 'kondisi janin dalam keadaan mencurigakan'
    else :
        prediksi = 'kondisi janin dalam keadaan tidak normal'
st.success(prediksi)