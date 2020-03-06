import numpy as np
from keras.models import load_model
import sys

def predict(messages, model_filepath="model.h5"):
    model = load_model(model_filepath)
    pred = model.predict(messages)
    return pred

if __name__ == "__main__":
    args = sys.argv
    if(len(args) != 2):
        print("python predict.py 'message'")
        sys.exit()
    messages = args[1]
    print(messages)
    message = [ord(x) for x in messages.strip()]
    message = message[:200]
    if len(message) < 30:
        exit("too short!!")
    if len(message) < 200:
        message += ([0] * (200 - len(message)))
    pred = predict(np.array([message]))
    predict_result = pred[0][0]
    print("spam: {:.2f}%".format(predict_result * 100))
