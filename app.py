import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from flask import Flask, render_template, request
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('captions.pkl', 'rb') as f:
    all_captions = pickle.load(f)


for i in range(len(all_captions)):
    caption = all_captions[i]
    caption = caption.lower()
    caption = caption.replace('[^A-Za-z]', '')
    caption = caption.replace('\s+', ' ')
    caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
    all_captions[i] = caption


print(all_captions[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
#print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1

app = Flask(__name__)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer) 
        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text


@app.route('/', methods=["POST", "GET"])
def index():  # put application's code here
    if request.method == "POST":
        img = request.files["myimage"]
        img.save('img1.png')
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        image = Image.open("img1.png")
        image = image.resize((224,224))
        features = {}
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        
        finalmodel = load_model("best_model.h5")
        
        y_pred = predict_caption(finalmodel, feature, tokenizer, 35)
        y_pred = y_pred.replace("startseq","")
        y_pred = y_pred.replace(" endseq",".")
        caption = [{'cap1':y_pred}]
        print(y_pred)
        return render_template("new.html", caption = caption)
    else:
        print("yes")
        return render_template("mainpage.html")


if __name__ == '__main__':
    app.debug = True
    app.run(port=3000)
