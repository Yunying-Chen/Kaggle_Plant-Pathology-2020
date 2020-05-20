import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet import MobileNet
import os
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model


def load_csv(path, flag="Training"):
    data = pd.read_csv(path)
    ids = data.iloc[:,0]
    if flag=="Training":
        labels = data.iloc[:, 1:].to_numpy()
        return ids,labels
    else:
        return ids

def load_imgs(path,imgs):
    imgs_data=[]
    for img in imgs:
        img_path = os.path.join(path,img+'.jpg')
        img = cv2.imread(img_path)
        image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        imgs_data.append(image)
    data = np.array(imgs_data, dtype=np.float32)
    data = data / 255
    return data

parser = argparse.ArgumentParser()
parser.add_argument("-train", "--train_paths", type=str,
                    default="train.csv",
                    help="The path of training data file.")
parser.add_argument("-test", "--test_paths", type=str,
                    default="test.csv",
                    help="The path of testing data file.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument("-output", "--output_path", type=str, default="submission.csv",
                    help="Learning rate.")
parser.add_argument("-e", "--epochs", type=int, default=1,
                    help="Num of epochs to train.")
parser.add_argument("-model", "--load_model", type=str, default=None,
                    help="Pretrained model.")

args = parser.parse_args()

train_ids,train_label = load_csv(args.train_paths)
train_data = load_imgs('images', train_ids)
x_train,x_val,y_train,y_val = train_test_split(train_data,train_label,test_size=0.1,random_state=2)
print("x_train:",x_train.shape)
print("y_train:",y_train.shape)

datagen = ImageDataGenerator(rotation_range=90,
                             shear_range=.15,
                              zoom_range=.15,
                              width_shift_range=.15,
                              height_shift_range=.15,
                              rescale=1/255,
                              brightness_range=[.5,1.5],
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest'
                              )

model = MobileNet(classes=4, weights=None, classifier_activation='softmax')
if args.load_model is not None:
    model = tf.keras.models.load_model(args.load_model)
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'] )
callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs/",histogram_freq=1)
]
model.fit_generator(datagen.flow(x_train, y_train),epochs=args.epochs, callbacks=callbacks, validation_data=(x_val,y_val))
model.save("model.h5")

test_ids = load_csv(args.test_paths,"Testing")
test_data = load_imgs('images',test_ids)

y_pred = model.predict(test_data)
res = pd.DataFrame()
res['image_id'] = test_ids
res['healthy'] = y_pred[:, 0]
res['multiple_diseases'] = y_pred[:, 1]
res['rust'] = y_pred[:, 2]
res['scab'] = y_pred[:, 3]
res.to_csv(args.output_path, index=False)