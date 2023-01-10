#%% 
#   Import Library
import os, re, json, pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


#   Path Variables
SAVED_MODEL_PATH = os.path.join(os.getcwd(),'saved_models')
TOKENIZER_PATH = os.path.join(SAVED_MODEL_PATH,'tokenizer.json')
MODEL_PATH = os.path.join(SAVED_MODEL_PATH,'model.h5')
OHE_PATH = os.path.join(SAVED_MODEL_PATH,'ohe.pkl')

#   Functions
def clean_text(text):
    '''
        This function take in a string and clean it using defined regex pattern and return it
    '''
    return re.sub(r'\d{2,4}s|\(\w{1,8}\)|trillion (yen|won)|\W\d+(m|bn)| [a-zA-Z]{1,2} |[^a-zA-Z]',' ',text).lower()


#%% 
# Data loading / Data Input
new_text = '''
Messi returns to the French club at an important time, with the team having lost for the first time this season against Lens last weekend. PSG's lead at the top of the Ligue 1 table has been cut to just four points as a result. The team's next game is against Chatearoux in the French Cup on Friday  a game that may come too soon for Messi.
'''
new_text = [new_text]

#%% 
# Data cleaning

#need to remove punctuations and HTML tags and to convert into lowercase
for index, data in enumerate(new_text):
    new_text[index] = clean_text(data)

#%% 
# Data preprocessing

#Load tokenizer
with open(TOKENIZER_PATH,'r') as f:
    loaded_tokenizer = json.load(f)

loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
new_text = loaded_tokenizer.texts_to_sequences(new_text)

# paddign sequences
new_text = pad_sequences(new_text, maxlen=200, padding='post',truncating='post')

# %% 
# Model Deployment

# to load the saved model
loaded_model = load_model(MODEL_PATH)
# loaded_model.summary()rese

# to load ohe model
with open(OHE_PATH,'rb') as f:
    loaded_ohe = pickle.load(f)

# make prediction
output = loaded_model.predict(new_text)

# display results
results = pd.DataFrame(output,index=['new_text'],columns=np.char.capitalize(list(loaded_ohe.categories_[0])))
print(results)
print('\nPredicted Category:', loaded_ohe.inverse_transform(output)[0][0].capitalize())
