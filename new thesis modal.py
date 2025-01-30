## new file 
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

import seaborn as sns
df = pd.read_csv(r'c:\Users\Lenovo\Downloads\Epileptic Seizure Recognition.csv')
df.head()
df.info()
df.tail()
df.describe()

df.astype
cols = df.columns
df.describe()

feature_columns = [col for col in df.columns if col.startswith('X')]
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Verify the conversion
print(df.dtypes)
df = df.drop(columns=['id'])

sample_indices = [0, 1, 2, 3]  # Change these indices as needed

plt.figure(figsize=(14, 10))

for idx in sample_indices:
    eeg_data = df.iloc[idx][feature_columns]
    plt.plot(eeg_data, label=f'Sample {idx}')

plt.title('EEG Data for Multiple Samples')
plt.xlabel('Time (Data Points)')
plt.ylabel('EEG Signal Amplitude')
plt.legend()
plt.grid(True)
plt.show()

tgt = df['y']
tgt[tgt > 1] = 0
import seaborn as sns
# Plot Seizure vs Non-Seizure distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=tgt, palette="Set2")
plt.title('Distribution of Seizure vs Non-Seizure Cases')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.show()

# Display counts
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)

X = df.iloc[:,1:-1].values
X.shape
y = df.iloc[:,-1:].values
y[y>1] = 0
y.shape

df.isnull().sum()


from sklearn.preprocessing import normalize, StandardScaler

X = df.drop('y', axis=1)
y = df['y']
df = pd.DataFrame(normalize(X))
# Initialize the counters for detected and managed outliers
detected_outliers = 0
managed_outliers = 0
# Loop through each of the 178 explanatory variables and calculate the IQR and bounds
for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Identify any data points that fall outside the bounds and either remove or adjust them
outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
if outliers.any():
        detected_outliers += outliers.sum()

        
        df.loc[outliers, col] = np.nanmedian(df[col])
        managed_outliers += outliers.sum()

print(f"Detected {detected_outliers} outliers and managed {managed_outliers} outliers.")

df['y'] = y

print('Normalized Totall Mean VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().mean()).mean()))
print('Normalized Totall Std VALUE for Epiletic: {}'.format((df[df['y'] == 1].describe().std()).std()))

print('Normalized Totall Mean VALUE for NOT Epiletic: {}'.format((df[df['y'] == 0].describe().mean()).mean()))
print('Normalized Totall Std VALUE for NOT Epiletic: {}'.format((df[df['y'] == 0].describe().std()).std()))

df.head()

import imblearn
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X, y = oversample.fit_resample(df.drop('y', axis=1), df['y'])

X.shape, y.shape
df.corr()
fig, ax = plt.subplots(figsize=(25, 25))

# Create heatmap
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.show()

print('Number of records of Non Epileptic {0} VS Epilepttic {1}'.format(len(y == True), len(y == False)))
df.head()
df.isnull().sum()


##split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Check the shapes after splitting
he = X_train, X_test, y_train, y_test
[arr.shape for arr in he]
X.isnull().sum()

# lets do ML
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of models to evaluate
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(),
          RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]

# Check models' classification reports
def evaluate_models(models):
    for model in models:
        model_name = type(model).__name__
        print(f"Training {model_name}...\n")
        
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Print classification report
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
        print("=" * 60)

# Run the evaluation
evaluate_models(models)

# Example of using iloc for numerical index-based access
X = df.iloc[:, 1:-1]  # Assuming the EEG features are in columns 1 to -1
y = df.iloc[:, -1]    # Assuming the target 'y' is the last column
feature_columns = [col for col in df.columns if col.startswith('X')]
print("Feature columns found:", feature_columns)

################preprocessing with feature col
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r'c:\Users\Lenovo\Downloads\Epileptic Seizure Recognition.csv')

# Print column names for verification
print("Column names in the DataFrame:")
print(df.columns)

# Define feature columns and target column
feature_columns = [col for col in df.columns if col.startswith('X')]
target_column = 'y'

# Convert columns to numeric, if needed
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

# Drop 'id' column if present
df = df.drop(columns=['id'], errors='ignore')

# Normalize features
X = df[feature_columns].values
y = df[target_column].values

# Handle missing values by replacing with median or another strategy if needed
df = pd.DataFrame(normalize(X), columns=feature_columns)
df[target_column] = y

# Apply outlier management
detected_outliers = 0
managed_outliers = 0

for col in feature_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    if outliers.any():
        detected_outliers += outliers.sum()
        df.loc[outliers, col] = np.nanmedian(df[col])
        managed_outliers += outliers.sum()

print(f"Detected {detected_outliers} outliers and managed {managed_outliers} outliers.")

# Re-check normalization results
print('Normalized Total Mean Value for Epileptic:', df[df[target_column] == 1].describe().mean().mean())
print('Normalized Total Std Value for Epileptic:', df[df[target_column] == 1].describe().std().std())
print('Normalized Total Mean Value for Non-Epileptic:', df[df[target_column] == 0].describe().mean().mean())
print('Normalized Total Std Value for Non-Epileptic:', df[df[target_column] == 0].describe().std().std())

# Perform oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(df[feature_columns], df[target_column])

# Plot the distribution of classes after resampling
plt.figure(figsize=(8, 6))
sns.countplot(x=y_resampled, palette="Set2")
plt.title('Distribution of Seizure vs Non-Seizure Cases After Resampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Seizure', 'Seizure'])
plt.show()

# Print class counts
print('Number of records for Non-Epileptic:', (y_resampled == 0).sum())
print('Number of records for Epileptic:', (y_resampled == 1).sum())

# Display correlation heatmap
fig, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Display shapes of splits
print(f"Shapes of the splits: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# Define models to evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = [LogisticRegression(), SVC(), DecisionTreeClassifier(),
          RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]

# Evaluate models
def evaluate_models(models):
    for model in models:
        model_name = type(model).__name__
        print(f"Training {model_name}...\n")
        
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Print classification report
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
        print("=" * 60)

# Run the evaluation
evaluate_models(models)

##########trying kfold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Define K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []

# K-Fold cross-validation
for train_index, test_index in kf.split(X_resampled):
    X_train_k, X_test_k = X_resampled[train_index], X_resampled[test_index]
    y_train_k, y_test_k = y_resampled[train_index], y_resampled[test_index]
    
    # Train the model
    model.fit(X_train_k, y_train_k)
    
    # Predict on test fold
    y_pred_k = model.predict(X_test_k)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_k, y_pred_k)
    accuracies.append(accuracy)

# Output the K-Fold results
if accuracies:
    print(f"K-Fold Accuracies: {accuracies}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
else:
    print("No accuracies available.")

# Define StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# Stratified K-Fold cross-validation
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train_k, X_test_k = X_resampled[train_index], X_resampled[test_index]
    y_train_k, y_test_k = y_resampled[train_index], y_resampled[test_index]
    
    model.fit(X_train_k, y_train_k)
    y_pred_k = model.predict(X_test_k)
    accuracy = accuracy_score(y_test_k, y_pred_k)
    accuracies.append(accuracy)

print(f"StratifiedKFold Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

X = df.iloc[:,1:-1].values
X.shape
y = df.iloc[:,-1:].values
y[y>1] = 0
y.shape

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True)
X_train.shape,y_test.shape

def denseBlock(dims,inp) :
    x = tf.keras.layers.BatchNormalization() (inp)
    x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Dense(64,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    return x
inp = tf.keras.layers.Input(shape=(177,),name='input')
x1 = denseBlock(256,inp)
x2 = denseBlock(512,inp)
x3 = denseBlock(1024,inp)
x = tf.keras.layers.Concatenate()([x1,x2,x3])
x = tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
out = tf.keras.layers.Dense(1,activation='sigmoid',name='output') (x)

model = tf.keras.models.Model(inp,out)
model.summary()

tf.keras.utils.plot_model(model,show_shapes=True)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(X_train)
x_test_scaled = sc.transform(X_test)
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50,batch_size=128,validation_split=0.2)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# BILSTM
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# Define the BiLSTM model
def create_bilstm_model(input_shape):
    inp = tf.keras.layers.Input(shape=input_shape, name='input')
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model


# Prepare the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model = create_bilstm_model((X_train_scaled.shape[1], 1))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

# Reshape input data for LSTM layer
X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=128, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###########done
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Define the convolutional front-end
def create_conv_frontend(input_shape):
    inp = Input(shape=input_shape, name='input')
    
    # Local Context CNN Encoder
    local_x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inp)
    local_x = MaxPooling1D(pool_size=2)(local_x)
    local_x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(local_x)
    local_x = MaxPooling1D(pool_size=2)(local_x)
    
    # Global Context CNN Encoder
    global_x = Conv1D(filters=64, kernel_size=15, activation='relu', padding='same')(inp)
    global_x = MaxPooling1D(pool_size=2)(global_x)
    global_x = Conv1D(filters=24, kernel_size=15, activation='relu', padding='same')(global_x)
    global_x = MaxPooling1D(pool_size=2)(global_x)
    
    # Concatenate local and global features
    x = concatenate([local_x, global_x], axis=-1)
    
    return inp, x

# Define the LSTM-based RNN-T model with regularization
def create_rnnt_model(input_shape):
    inp, conv_features = create_conv_frontend(input_shape)
    
    # Add LSTM layers
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(conv_features)
    x = Dropout(0.4)(x)
    x = LSTM(64, kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    
    # Output layer
    out = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model

# Prepare the data
X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)

# Create the model
model = create_rnnt_model((X_train_scaled.shape[1], 1))

# Compile the model with Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Define callbacks for checkpointing and early stopping
checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss', mode='min')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    batch_size=128, 
                    validation_split=0.2, 
                    callbacks=[checkpoint_cb, early_stopping_cb])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
