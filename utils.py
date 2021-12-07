import matplotlib.pyplot as plt
import librosa
import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.utils import to_categorical

max_pad_len = 174
num_rows = 40
num_columns = 174
num_channels = 1


def draw_spectrogram(file_name):
    librosa_audio, librosa_rate = librosa.load(file_name)
    plt.figure(figsize=(12, 4))
    plt.plot(librosa_audio)
    return plt


def load_model(model):
    try:
        # Load the model
        json_file = open('models/cnn_tunned.json')
        json_model = json_file.read()
        json_file.close()

        # Load the weights
        model = model_from_json(json_model)
        model.load_weights('models/cnn_tunned.h5')
        print(f'Model {model} loaded! ')
        return model
    except Exception as e:
        print(f'Error cargando el modelo {model}: {e}')
        return None


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs


def predict(file, type):
    probabilities = pd.DataFrame(columns=['Category', 'Prob'])
    model = cnn if type == 'cnn' else mlp
    prediction_feature = extract_features(file)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict(prediction_feature)
    classes_x = np.argmax(predicted_vector, axis=1)
    predicted_class = le.inverse_transform(classes_x)
    # print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        probabilities.loc[len(probabilities)] = [category[0], format(predicted_proba[i], '.32f')]
        # print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))

    return predicted_class[0], probabilities


try:
    # Load CNN model
    cnn = load_model('cnn')

    # Load MLP model
    mlp = load_model('mlp')

    # Load features
    features_df = pd.read_csv('Resources/features_df.csv')
    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
except Exception as e:
    print(f'Error loading the models: {e}')
