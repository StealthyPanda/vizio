import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow import keras
from codex import dblog, ok, warning, error


def gpustatus():
    from codex import dblog, ok, warning
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        ok('Using GPU')
        dblog('Found these GPU devices:', '\n', gpu)
    else:
        warning("No GPU found!")
gpustatus()

model = tf.keras.models.load_model('decoder.keras')

st.set_page_config(
    layout = 'wide'
)
left, _, right = st.columns([2, 1, 7])

left.markdown("### Model Inputs")
inputs = [left.slider(f'Input {i}', -10., 10., 0., step = 0.0001) for i in range(model.input_shape[-1])]

def getplotoutput(inputs, parent):
    output = np.reshape(model(np.array(inputs, dtype = np.float32).reshape(1, model.input_shape[-1])), (28, 28))
    parent.plotly_chart(px.imshow(output, height = 700, width = 700, color_continuous_scale='viridis'))

right.markdown(f"### Model Output")
getplotoutput(inputs, right)

