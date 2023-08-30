from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import form_data
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

x, y = form_data.get_df()
x = form_data.get_normalized(x)
x_train, x_test, y_train, y_test = form_data.get_dataset(x, y)

def build_model():
    model = keras.Sequential([
        layers.Dense(units=32, activation="relu"),
        layers.Dense(units=64, activation="relu"),
        layers.Dense(units=32, activation="relu"),
        layers.Dense(units=1, activation="sigmoid"),
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

def show_history_loss(history):
    plt.plot(history.history['loss'], label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def confusion_mat(labels, predicts):
    return confusion_matrix(labels, predicts)

def roc(labels, predictions):
    # ROC 분석 수행
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # ROC 곡선 그리기
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

model = build_model()
history = model.fit(x_train, y_train, epochs=150, batch_size=512)
loss, acc = model.evaluate(x_test, y_test)

print(acc)

predictions = model.predict(x_test)
predicted_labels = np.round(predictions)
#roc(y_test, predicted_labels)
#print(confusion_mat(y_test, predicted_labels))
