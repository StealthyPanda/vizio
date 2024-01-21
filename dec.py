import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
from codex import dblog, ok, warning, error



st.set_page_config(
    layout = 'wide'
)

def gpustatus():
    from codex import dblog, ok, warning
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
        ok('Using GPU')
        st.info("Using GPU")
        dblog('Found these GPU devices:', '\n', gpu)
    else:
        st.warning("Not using GPU!")
        warning("No GPU found!")

gpustatus()
# if 'gpushown' not in st.session_state:
#     gpustatus()
#     st.session_state['gpushown'] =  True

model = tf.keras.models.load_model('decoder.keras')

left, _, right = st.columns([2, 1, 7])

left.markdown("### Model Inputs")
mult = left.number_input('Multiplier:', value = 1.)
defvals = [6.22, 5.66, 5.61, 8.52, 7.79]
inputs = [left.slider(f'Input {i}', 0., 10., defvals[i], step = 0.0001) * mult for i in range(model.input_shape[-1])]

def getplotoutput(inputs, parent):
    output = np.reshape(model(np.array(inputs, dtype = np.float32).reshape(1, model.input_shape[-1])), (28, 28))
    parent.plotly_chart(px.imshow(output, height = 700, width = 700, color_continuous_scale='viridis'))

right.markdown(f"### Model Output")
getplotoutput(inputs, right)

