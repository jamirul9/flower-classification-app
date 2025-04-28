from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = load_model('model/flower_model.h5')
classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    confidence = 0
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)[0]
            prediction_index = np.argmax(predictions)
            prediction = classes[prediction_index]
            confidence = round(predictions[prediction_index]*100, 2)

            return render_template('index.html', prediction=prediction, confidence=confidence, image_path=filepath)

    return render_template('index.html')

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
