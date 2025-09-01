
# app.py
import streamlit as st
import numpy as np
from pathlib import Path
from loader import load_quadruple_txt
from nn import forwardPropTest
import pandas as pd, altair as alt

st.set_page_config(page_title="SimpleNN • MNIST Dev Demo", page_icon="🧠", layout="centered")
st.title("🧠 SimpleNN • Dev Set Demo")

APP_DIR = Path(__file__).parent          # directory where app.py lives
WEIGHTS = APP_DIR  / "nn_weights.txt"   # <-- use your real filename
DEVSET  = APP_DIR / "dev.npz"

# ---- load assets ----
@st.cache_resource
def load_model():
    w1, b1, w2, b2, meta = load_quadruple_txt(WEIGHTS)
    return w1, b1, w2, b2, meta

@st.cache_data
def load_dev():
    d = np.load(DEVSET)
    X_dev, Y_dev = d["X_dev"], d["Y_dev"]  # expect X_dev=(784,N), Y_dev=(N,)
    # If your X_dev is (N,784), transpose it:
    if X_dev.shape[0] != 784 and X_dev.shape[1] == 784:
        X_dev = X_dev.T
    return X_dev, Y_dev

missing = []
if not WEIGHTS.exists(): missing.append(f"Missing `{WEIGHTS}`")
if not DEVSET.exists():  missing.append(f"Missing `{DEVSET}`")
if missing:
    st.error(" • ".join(missing))
    st.stop()

w1, b1, w2, b2, meta = load_model()
X_dev, Y_dev = load_dev()
N = X_dev.shape[1]
classes = meta.get("class_names") or list(range(int(meta.get("output_dim", 10))))
preproc = meta.get("preproc")  # optional
idx = int(np.random.randint(0, N))

# ---- controls ----
st.sidebar.header("Controls")
#seed = st.sidebar.number_input("Random seed", value=0, step=1)
#np.random.seed(int(seed))
# your Kaggle code used randint(1,800); more robust is 0..N-1:
#default_idx = int(np.random.randint(0, N))
#idx = st.sidebar.slider("Index", 0, N-1, default_idx)
if st.sidebar.button("🎲 Randomize"):
    idx = int(np.random.randint(0, N))

# ---- pick sample & show image ----
x_col = X_dev[:, idx:idx+1]              # (784,1)
img    = X_dev[:, idx].reshape(28, 28)   # for display
label  = int(Y_dev[idx]) if Y_dev.ndim == 1 else int(Y_dev[idx, 0])

col1, col2 = st.columns([1, 1.2])
with col1:
    # normalize image for display only
   
    vmin, vmax = 0.0, 1.0 if img.max() <= 1.0 else 255.0
    st.image((img - vmin) / (vmax - vmin + 1e-8), width=196, caption=f"True label: {label}")
with col2:
    pred, probs, logits = forwardPropTest(w1, w2, b1, b2, x_col)
    
    st.markdown(f"**Prediction:** `{classes[pred]}`")
    st.markdown(f"Confidence: **{float(probs[pred]):.3f}**")
    #st.bar_chart(probs, height=220)


   

df = pd.DataFrame({"class": [str(c) for c in classes], "prob": probs})
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("class:N", title="Class",
                axis=alt.Axis(labelAngle=0))   # 0 = horizontal; try 45 or -45
        ,
        y=alt.Y("prob:Q", title="Probability",
                scale=alt.Scale(domain=[0, min(1, 0.2 + max(probs))])),
        tooltip=["class", alt.Tooltip("prob:Q", format=".3f")]
    )
    .properties(height=220)
)
st.altair_chart(chart, use_container_width=True)


#st.caption(f"idx={idx} • input_dim={X_dev.shape[0]} • K={len(classes)} • hidden_dim={meta.get('hidden_dim','?')}")
