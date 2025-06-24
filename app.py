import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode
from codice.dct_utils import dct_base, dct2, idct2 

app = Flask(__name__)
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="BMP")
    return b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img_b64 = None
    processed_img_b64 = None

    if request.method == 'POST':
        file = request.files.get('image_file')
        if file:
            F = int(request.form.get('block_size'))  # Dimensione del blocco DCT
            d = int(request.form.get('threshold'))
  

            # Carica e prepara immagine
            image = Image.open(file).convert('L')
            img = np.array(image).astype(np.float32)

            h, w = img.shape
            h -= h % F
            w -= w % F
            img = img[:h, :w]
            img -= 128

            # Prepara matrice DCT
            D = dct_base(F)

            # Applica DCT a blocchi
            for i in range(0, h, F):
                for j in range(0, w, F):
                    block = img[i:i+F, j:j+F]
                    img[i:i+F, j:j+F] = dct2(block, D)

            # Elimina alte frequenze
            mask = np.zeros((F, F))
            for i in range(F):
                for j in range(F):
                    if i + j < d:
                        mask[i, j] = 1

            for i in range(0, h, F):
                for j in range(0, w, F):
                    img[i:i+F, j:j+F] *= mask

            # Applica IDCT a blocchi
            for i in range(0, h, F):
                for j in range(0, w, F):
                    block = img[i:i+F, j:j+F]
                    img[i:i+F, j:j+F] = idct2(block, D)

            img += 128
            img = np.clip(img, 0, 255).astype(np.uint8)
            final_image = Image.fromarray(img)

            # Converti in base64 per visualizzazione
            original_img_b64 = image_to_base64(image)
            processed_img_b64 = image_to_base64(final_image)

    return render_template('index.html',
                           original_image=original_img_b64,
                           processed_image=processed_img_b64)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
