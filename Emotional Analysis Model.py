import pandas as pd 
import numpy as np 
import tensorflow as Tf  
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.layers import LSTM, Dense, Embedding 
from tensorflow.keras.models import Sequential 

data = pd.read_csv('emotion.csv')

new_data = data.drop(columns = 'Unnamed: 0', axis = 1)

text = new_data['text'].tolist()
emotion= new_data['Emotion'].tolist()

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
pad_data = pad_sequences(sequence, maxlen = 100)


labels = pd.get_dummies(new_data['Emotion'])

model = Sequential([
    Embedding(input_dim = 5000, output_dim = 100, input_length =100),
    LSTM(units = 100),
    Dense(13, activation = 'softmax')
])


model.compile(loss= 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
x_train, x_test, y_train, y_test = train_test_split(pad_data, labels, test_size = 0.2, random_state = 42)

print(x_train.shape, x_test.shape, y_test.shape, y_train.shape)
model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), batch_size= 32)


def predict_data(model, tokenizer, text, max_len):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen = max_len)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction[0])
    return predicted_label 


txt_input = 'I love you'
predicted_label = predict_data(txt_input)
print(f"Prediction Emoton Label: {predicted_label}")


