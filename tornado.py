import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
class CAPEAmplifier(tf.keras.layers.Layer):
    def __init__(self, threshold=2000, scale=0.001, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.scale = scale

    def call(self, inputs):
        cape = inputs[:, 1]
        boost = tf.sigmoid((cape - self.threshold) * self.scale)
        mod = 1.0 + 0.3 * boost
        return tf.expand_dims(mod, axis=-1)

@register_keras_serializable()
class LCLSuppressor(tf.keras.layers.Layer):
    def __init__(self, threshold=1400, scale=0.002, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.scale = scale

    def call(self, inputs):
        lcl = inputs[:, 2]
        suppression = tf.sigmoid((lcl - self.threshold) * self.scale)
        return tf.expand_dims(1.0 - 0.25 * suppression, axis=-1)

@register_keras_serializable()
class STPActivator(tf.keras.layers.Layer):
    def __init__(self, threshold=1.5, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.scale = scale

    def call(self, inputs):
        stp = inputs[:, 4]
        activation = tf.sigmoid((stp - self.threshold) * self.scale)
        return tf.expand_dims(1.0 + 0.2 * activation, axis=-1)

@register_keras_serializable()
class ModulationMixer(tf.keras.layers.Layer):
    def call(self, inputs):
        cape_mod, lcl_mod, stp_mod = inputs
        combined = cape_mod * lcl_mod * stp_mod
        return 1.0 + 0.3 * tf.tanh(combined - 1.0)

@register_keras_serializable()
def trust_activation(x):
    return 0.5 + tf.sigmoid(x)

tornado_model = load_model("models/TornadoNet.h5", custom_objects={
    'CAPEAmplifier': CAPEAmplifier,
    'LCLSuppressor': LCLSuppressor,
    'STPActivator': STPActivator,
    'ModulationMixer': ModulationMixer
})
trust_model = load_model("models/TornadoTrustNet.h5", custom_objects={
    'mse': tf.keras.losses.MeanSquaredError(),
    'trust_activation': trust_activation
})
scaler = joblib.load("models/TornadoTrustScaler.pkl")

scenarios = [
    {
        "label": "Classic Supercell Day",
        "features": [340, 3100, 820, 22, 2.9],
        "expected": "Tornado"
    },
    {
        "label": "High LCL, Dry Thunderstorm",
        "features": [190, 2700, 1750, 16, 0.8],
        "expected": "No Tornado"
    },
    {
        "label": "Marginal Shear, High Instability",
        "features": [240, 3000, 950, 12, 1.6],
        "expected": "Possibly Tornado"
    },
    {
        "label": "Cool Humid Blip",
        "features": [210, 950, 680, 8, 0.6],
        "expected": "No Tornado"
    },
    {
        "label": "STP-Peaked Composite Storm",
        "features": [320, 2800, 870, 20, 3.1],
        "expected": "Tornado"
    },
    {
        "label": "Low Vorticity High CAPE",
        "features": [270, 3200, 1000, 10, 0.9],
        "expected": "Edge Case"
    }
]

# --- Classification logic ---
def classify(pred, trust_score):
    if pred < 0.4:
        return "No Tornado"
    elif 0.4 <= pred <= 0.55:
        if trust_score > 1.0:
            return "Tornado"
        elif trust_score < 0.8:
            return "No Tornado"
        else:
            return "Possibly Tornado"
    else:
        return "Tornado"

# --- Evaluation ---
print("\n🌪️ TornadoNet + TornadoTrustNet Dual Evaluation:\n")
for case in scenarios:
    features = case["features"]
    raw = np.array(features, dtype="float32").reshape(1, -1)

    scaled = scaler.transform(pd.DataFrame([features], columns=[
        "storm_relative_helicity", "CAPE",
        "lifted_condensation_level", "bulk_wind_shear", "significant_tornado_param"
    ]))

    base_pred = tornado_model(raw).numpy()[0][0]
    trust_score = trust_model(scaled).numpy()[0][0]
    verdict = classify(base_pred, trust_score)

    print(f"{case['label']}")
    print(f"  Features      : {features}")
    print(f"  TornadoNet    : {base_pred:.2f}")
    print(f"  Trust Score   : {trust_score:.2f}")
    print(f"  Final Verdict : {verdict}")
    print(f"  Expected      : {case['expected']}\n")