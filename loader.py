import numpy as np
import json

def load_quadruple_txt(path):
    """
    Load weights from the text (JSON) file created above.
    Returns (w1, b1, w2, b2)
    """
    def unpack(obj):
        arr = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"])

    with open(path, "r") as f:
        payload = json.load(f)

    w1 = unpack(payload["w1"])
    b1 = unpack(payload["b1"])
    w2 = unpack(payload["w2"])
    b2 = unpack(payload["b2"])
    meta = payload.get("meta", {})
    return w1, b1, w2, b2, meta