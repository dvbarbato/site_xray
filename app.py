import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

import sklearn

st.write('este Meu SITE!!!!!')


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


st.title('Classificar')


arquivo = st.file_uploader("Upload de uma imagem",
                           type=["jpeg"])


if arquivo is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')
    test_image = Image.open(arquivo)
    st.image(test_image, caption="Input Image", width=200)
    test_image.save('img.jpeg')

    img_tratada = imagem('img.jpeg')

    previsao = network1_loaded.predict(img_tratada)

    st.write(previsao)

# # scaler = joblib.load('scaler_minmax.pkl')


# import cv2
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

# model = tf.keras.models.load_model("weights1.hdf5")
# # load file
# uploaded_file = st.file_uploader("Choose a image file", type="jpeg")

# map_dict = {0: 'pneumonia',
#             1: 'normal'}


# if uploaded_file is not None:
#     # Convert the file to an opencv image.
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)
#     opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(opencv_image, (128, 128))
#     # Now do something with the image! For example, let's display it:
#     st.image(opencv_image, channels="RGB")

#     resized = mobilenet_v2_preprocess_input(resized)
#     img_reshape = resized[np.newaxis, ...]

#     Genrate_pred = st.button("Generate Prediction")
#     if Genrate_pred:
#         prediction = model.predict(img_reshape).argmax()
#         st.title("Predicted Label for the image is {}".format(
#             map_dict[prediction]))
