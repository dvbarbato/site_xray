import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import time

import sklearn

# st.write('este Meu SITE!!!!!')


with open('network1.json') as json_file:
    json_saved_model = json_file.read()

network1_loaded = tf.keras.models.model_from_json(json_saved_model)
network1_loaded.load_weights('weights1.hdf5')
network1_loaded.compile(loss='binary_crossentropy',
                        optimizer='Adam', metrics=['accuracy'])

scaler = joblib.load('scaler_minmax.pkl')


def imagem(uma_imagem, scaler=scaler):
    st.write(uma_imagem)
    imagem = cv2.imread(uma_imagem)
    (h, w) = imagem.shape[:2]

    imagem = cv2.resize(imagem, (128, 128))
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    imagem = imagem.ravel()
    # print(imagem.shape)
    imagem = scaler.transform(imagem.reshape(1, -1))
    return imagem


st.title('Classificador de Imagens')
st.markdown('**Obs:** Valores menores que 0.5 indica uma possível inflamação no pulmão. Valores entre 0.5 e 1 indica normalidade')

arquivo = st.file_uploader("Faça o Upload da imagem",
                           type=["jpeg"])


if arquivo is None:
    st.text('Esperando pelo upload....')
else:
    slot = st.empty()

    with st.spinner('Carregando imagem...'):
        time.sleep(3)

    # slot.text('Executando...')
    test_image = Image.open(arquivo)
    st.image(test_image, caption="Input Image", width=200)
    test_image.save('img.jpeg')

    img_tratada = imagem('img.jpeg')

    previsao = network1_loaded.predict(img_tratada)

    # st.write(previsao[0][0])

    if previsao > 0.5:
        output = 'PULMÃO NORMAL '
        st.write("A pontuação de similaridade é de aproximadamente: ",
                 previsao[0][0])
        st.success(output, icon="✅")
    else:
        output = 'Possivel INFLAMAÇÃO'
        st.write("A pontuação de similaridade é de aproximadamente: ",
                 previsao[0][0])
        st.warning(output, icon="⚠️")
    slot.text('Concluido')
