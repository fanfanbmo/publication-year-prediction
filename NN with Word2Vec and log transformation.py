import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

'''Due to the inherent complexity of the algorithm and hardware limitations, 
it is difficult to reproduce the exact same result, 
but the difference between results of each run are controlled under 0.1.'''

seed_value = 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

train_path = "train.json"
test_path = "test.json"
train = pd.DataFrame.from_records(json.load(open(train_path))).fillna("")
test = pd.DataFrame.from_records(json.load(open(test_path))).fillna("")

# looked at the distribution of the variable year in the training set

plt.figure(figsize=(8, 6))
plt.boxplot((train['year'].astype(int)), vert=False)
plt.title('Boxplot for year')
plt.savefig("Boxplot for year.png")
plt.show(block=False)

# transformed the data into right skewed first, then used log transformation

train['year'] = train['year'].astype(int)
train['year_neg'] = max(train['year'])-train['year'] + 1
train['year_log'] = np.log1p(train['year_neg'])

train['year'].hist(bins= 20)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Year distribution')
plt.savefig("Year distribution.png")
plt.show(block=False)

train['year_log'].hist(bins= 20)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Year distribution after log')
plt.savefig("Year distribution after log.png")
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.boxplot((train['year_log']), vert=False)
plt.title('Boxplot for year after log')
plt.savefig("Boxplot for year after log.png")
plt.show(block=False)

# extracted year from title

train['extracted_year'] = train['title'].str.extract(r'(\b\d{4}\b)')

test['extracted_year'] = test['title'].str.extract(r'(\b\d{4}\b)')

# Filling missing values with 0 and converting the extracted years to integers.
new_feature_train = train['extracted_year'].fillna(0).astype(int)
new_feature_test = test['extracted_year'].fillna(0).astype(int)

print(new_feature_train.head())
print(new_feature_test.head())

# Dummy encoding categorical data
features_to_encode = ['ENTRYTYPE', 'publisher']
train_data_subset = train[features_to_encode]
test_data_subset = test[features_to_encode]

# These lines create subsets of the training and test data containing only the specified categorical features.
combined_data = pd.concat([train_data_subset, test_data_subset], axis=0)

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = pd.DataFrame(encoder.fit_transform(combined_data), columns=encoder.get_feature_names_out(features_to_encode))

train_dummy_encoded = encoded_data.iloc[:len(train)].reset_index(drop=True)
test_dummy_encoded = encoded_data.iloc[len(train):].reset_index(drop=True)

print(test_dummy_encoded.head())
print(train_dummy_encoded.head())

# Cleaning and encoding for text data
# The feature data are sparse
# Used W2V to control the dimensionality
# Did not include editor because most of the data are missing
# tried to use PCA to reduce dimensionality but caused higher MAE

def clean_and_tokenize(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
    return cleaned_text.split()

# First name and after name of each author were grouped together and seen as one word
train['title_tokens'] = train['title'].apply(clean_and_tokenize)
train['author_tokens'] = train['author'].apply(lambda x: [''.join(clean_and_tokenize(name)) for name in x])
train['author_tokens'] = train['author_tokens'].apply(lambda x: ' '.join(x))
train['author_tokens'] = train['author_tokens'].apply(clean_and_tokenize)
train['abstract_tokens'] = train['abstract'].apply(clean_and_tokenize)

train['combined_tokens'] = train['title_tokens'] + train['author_tokens'] + train['abstract_tokens']

test['title_tokens'] = test['title'].apply(clean_and_tokenize)
test['author_tokens'] = test['author'].apply(lambda x: [''.join(clean_and_tokenize(name)) for name in x])
test['author_tokens'] = test['author_tokens'].apply(lambda x: ' '.join(x))
test['author_tokens'] = test['author_tokens'].apply(clean_and_tokenize)
test['abstract_tokens'] = test['abstract'].apply(clean_and_tokenize)

test['combined_tokens'] = test['title_tokens'] + test['author_tokens'] + test['abstract_tokens']

total = pd.DataFrame()
total['combined_tokens'] = pd.concat([train['combined_tokens'], test['combined_tokens']], axis=0)

w2v_model = Word2Vec(sentences=total['combined_tokens'], vector_size=300, window=5, min_count=1, workers=4, seed=seed_value)


def get_embeddings(sequence):
    return np.mean([w2v_model.wv[word] for word in sequence], axis=0)

train['embeddings'] = train['combined_tokens'].apply(get_embeddings)

df_embeddings_train = pd.DataFrame(train['embeddings'].to_list(), columns=[f'embedding_{i}' for i in range(300)])


test['embeddings'] = test['combined_tokens'].apply(get_embeddings)

df_embeddings_test = pd.DataFrame(test['embeddings'].to_list(), columns=[f'embedding_{i}' for i in range(300)])

# Concatinate all the features

train_encoded = pd.concat([train_dummy_encoded, df_embeddings_train, new_feature_train], axis=1)
test_encoded = pd.concat([test_dummy_encoded, df_embeddings_test, new_feature_test], axis=1)
print(train_encoded.head())

y = train['year_log']
X_train, X_val, y_train, y_val = train_test_split(train_encoded, y, test_size=0.3, random_state=seed_value, stratify = y)

print('X_train size is: ', X_train.shape)
print('X_val  size is: ', X_val.shape)
print('y_train size is: ', y_train.shape)
print('y_val  size is: ', y_val.shape)


# use neural network with reduce learning rate and early stopping
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_absolute_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_val_scaled, y_val), callbacks=[reduce_lr, early_stopping])

mae = model.evaluate(X_val_scaled, y_val)
print(f"Mean Absolute Error on Validation Set: {mae}")

# decode the year for validation set
predictions = model.predict(X_val_scaled)
y_val_decoded = (max(train['year'])+1-np.expm1(y_val)).astype(int)
y_pred_decoded = (max(train['year'])+1-np.expm1(predictions)).astype(int)
mae_decoded = mean_absolute_error(y_val_decoded, y_pred_decoded)
print(f'Mean Absolute Error on Validation Set: {mae_decoded}')

#decode year for test set
test_scaled = scaler.fit_transform(test_encoded)
pred = pd.DataFrame(model.predict(test_scaled))

predicted = pd.DataFrame()
predicted['year'] =(max(train['year'])+1-np.expm1(pred)).astype(int)
predicted.to_json("predicted.json", orient='records', indent=2)
