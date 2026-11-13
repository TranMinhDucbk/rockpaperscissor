import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras
from tkinter import filedialog
from keras.preprocessing import image

model = keras.models.load_model("rps.keras")   # load model đã train, không train lại
uploaded = filedialog.askopenfilenames()     
for fn in uploaded:
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)