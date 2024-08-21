import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

DATA_DIR = 'data/labeled/images/'
def load_img(file=None):
    if file is None:
        file = np.random.choice(os.listdir(DATA_DIR))
    file = os.path.join(DATA_DIR, file)
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    return img

def show_img(file=None):
    if file is None or isinstance(file, str):
        img = load_img(file)
    else:
        img = file
    plt.imshow(img)
    plt.show()