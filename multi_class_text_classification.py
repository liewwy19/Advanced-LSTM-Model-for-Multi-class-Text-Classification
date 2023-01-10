"""
Created on Tue Jan 10 09:12:18 2023

@author: Wai Yip LIEW (liewwy19@gmail.com)
"""

# %% 
#   1. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, os, datetime, json, pickle, random

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

#   constant variables
SEED = 142857
SOURCE_URL = r'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
LOG_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SAVED_MODEL_PATH = os.path.join(os.getcwd(),'saved_models')

#   functions
def clean_text(text):
    '''
        This function take in a string and clean it using defined regex pattern and return it
    '''
    return re.sub(r'\d{2,4}s|\(\w{1,8}\)|trillion (yen|won)|\W\d+(m|bn)| [a-zA-Z]{1,2} |[^a-zA-Z]',' ',text).lower()

def clean_text_residual(text):
    '''
        This function further clean up those single or double letter words left over from previous cleaning step        
    '''
    return re.sub(r' [a-zA-Z]{1,2} ',' ',text)

# %%
#   2. Data Loading
df = pd.read_csv(SOURCE_URL)
# %%
#   3. Data Inspection
print(df.info(),end='\n'+'-'*20+'\n')
print('Category Distribution:')
print(df.category.value_counts(ascending=True),end='\n'+'-'*20+'\n')
print('Category Distribution (Normalized):')
print(df.category.value_counts(ascending=True,normalize=True),end='\n'+'-'*20+'\n')
print('Max. Words Count:',df['text'].apply(str.split).apply(len).max())

#   - check for duplicates
print('Duplicates: ',df.duplicated().sum())

# %%
#   - randomly select few text for inspection (repeat if needed)
for i in range(2):
    print('-'*20)
    idx = random.choice(df.index)
    print(f'Index: {idx} - {len(df["text"][idx].split())} words')
    print(df["text"][idx])
    
# %%
#   4. Data Cleaning
'''
    a. remove duplicates - 99 duplicates found
    b. remove monetory value (e.g. £35m, $180bn, trillion yen) df.iloc[15:18]
    c. remove decades/centuries (e.g. 1990s, 80s)
    d. remove words with lenght=1 e.g (i, a, s, t)
    e. remove abbreviation inside parenthesis e.g (ioc) (ifpi) df.iloc[2176]
    f. remove puntuations

'''
#   - remove duplicates
df.drop_duplicates(inplace=True)

#   - calling clean_text function to do text cleaning
df['text'] = df['text'].apply(clean_text)

#   - further clean up text before training
for _ in range(4):
    df['text'] = df['text'].apply(clean_text_residual)

#   - quick checking after data cleaning
print('Words Count for the longest text:',df['text'].apply(str.split).apply(len).max(),end='\n'+'-'*20+'\n')
print('Randomly selected text:\n',df['text'][random.randrange(len(df))])
# %%
#   5. Features selection
text = df['text']
category = df['category']
category_names = np.char.capitalize(list(category.unique()))
category_names.sort() # sort the list alphabetically so that it can be use later within the confusion matrix
nClass = len(category_names)

#%%
#   6. Data preprocessing
#   - Tokenizer
NUM_WORDS = 5000
OOV_TOKEN = '<OOV>'

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN) #instantiate the object

#   - to train the tokenizer
tokenizer.fit_on_texts(text)

#   - to transform the text using tokenizer
text = tokenizer.texts_to_sequences(text)

#   - Padding
padded_text = pad_sequences(text, maxlen=200, padding='post', truncating='post')

#   - One hot encoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category.values[::,None])

#   - expand dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text, axis=-1)

#   - perform train-test-split
X_train,X_test,y_train,y_test = train_test_split(padded_text,category,test_size=0.2,random_state=SEED)


# %%
#   7. Model Development
embedding_layer = 64

model = Sequential()
model.add(Embedding(NUM_WORDS,embedding_layer))
model.add(LSTM(embedding_layer,return_sequences=True)) 
model.add(Dropout(0.3)) # reduce the chance for overfitting
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(nClass, activation='softmax'))
model.summary()
plot_model(model, show_shapes=True)

# %%
#   8. Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 

#   9. Model Training
#   - define callback functions
tb = TensorBoard(log_dir=LOG_PATH)
es = EarlyStopping(monitor='val_loss',patience=8, verbose=1, restore_best_weights=True)
mc = ModelCheckpoint(os.path.join(SAVED_MODEL_PATH,'best_model_chkpt.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#   - model training
BATCH_SIZE = 64
EPOCHS = 50
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tb,es,mc])

# %%
#   10. Model analysis

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.legend(['Training','Validation'])
plt.show()

y_predicted = model.predict(X_test)

y_predicted = np.argmax(y_predicted, axis=1)
y_actual = np.argmax(y_test, axis=1)

#   - compute classification report
print(classification_report(y_actual,y_predicted,target_names=category_names))

#   - confusion matrix visualization
disp = ConfusionMatrixDisplay.from_predictions(y_actual, y_predicted,cmap='Blues',display_labels=category_names,xticks_rotation=45)
ax = disp.ax_.set(title='Confusion Matrix', xlabel='Predicted Category', ylabel='Actual Category')

#%%
#   11. Model saving

#   - to save trained model
model.save(os.path.join(SAVED_MODEL_PATH,'model.h5')) # save train model

#   - to save one hot encoder model
with open(os.path.join(SAVED_MODEL_PATH,'ohe.pkl'),'wb') as f:
    pickle.dump(ohe,f)

#   - to save tokenizer
token_json = tokenizer.to_json() # convert to json format 1st
with open(os.path.join(SAVED_MODEL_PATH,'tokenizer.json'),'w') as f:
    json.dump(token_json,f)

#%%
#   12. Model deployment

#   - Unseen text for prediction
new_text = '''
Messi returns to the French club at an important time, with the team having lost for the first time this season against Lens last weekend. PSG's lead at the top of the Ligue 1 table has been cut to just four points as a result. The team's next game is against Chatearoux in the French Cup on Friday  a game that may come too soon for Messi.
'''
new_text = [clean_text(new_text)]

#   - tokenized new_text
new_text = tokenizer.texts_to_sequences(new_text)

#   - padding sequences
new_text = pad_sequences(new_text, maxlen=200, padding='post',truncating='post')

#   - make prediction
output = model.predict(new_text)

#   - display results
results = pd.DataFrame(output,index=['new_text'],columns=np.char.capitalize(list(ohe.categories_[0])))
print(results)
print('\nPredicted Category:', ohe.inverse_transform(output)[0][0].capitalize())


# %%
