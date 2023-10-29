import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
dia_model = pickle.load(open('estimasi_diamond.sav','rb'))
image = Image.open('banner.jpeg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Harga Diamond')
st.info('Cut Diamond (Ideal = 1,Premium = 2,Good = 3,Very Good = 4,Fair = 5)', icon="ℹ️")
st.info('Color Diamond from D/Best to J/Worse (D = 1,E = 2,F = 3,G= 4,H = 5,I = 6,J = 7)', icon="ℹ️")
st.info('Clarity Diamond (1/Worst ~ 7/Best)', icon="ℹ️")

col1, col2,col3=st.columns(3)
with col1:
    carat = st.number_input('Input Nilai Carat Diamond :')
with col2:
    cut  = st.number_input('Input Nilai Cut Diamond :')
with col3:
    color  = st.number_input('Diamond Color :')
with col1:
    clarity = st.number_input('Input Clarity Diamond :')
with col2:
    depth = st.number_input('Depth% = z / mean(x,y) :')
with col3:
    table = st.number_input('Widest Point at top diamond :')
with col1:
    x = st.number_input('lenght :')
with col2:
    y = st.number_input('width :')
with col3:
    z = st.number_input('depth :')

#code untuk estimasi
ins_est=''

#membuat button
with col1:
    if st.button('Estimasi Harga'):
        dia_pred = dia_model.predict([[carat,cut,color,clarity,depth,table,x,y,z]])

        st.success(f'Estimasi Harga : {dia_pred[0]:.2f}')