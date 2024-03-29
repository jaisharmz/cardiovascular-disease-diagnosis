import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras import layers
import numpy as np

hidden_units = 64
dropout_rate = 0.4

layers_dimensions = [hidden_units,hidden_units,hidden_units,hidden_units,hidden_units,hidden_units,hidden_units,hidden_units,hidden_units]
initializer = tf.keras.initializers.GlorotNormal(seed=2) 
model_nn = keras.Sequential([
                            layers.BatchNormalization(input_shape=[18]),
                            layers.Dense(layers_dimensions[0],kernel_initializer=initializer,bias_initializer='zeros'), 
                            layers.BatchNormalization(),
                            layers.Activation("relu"),
                            layers.Dropout(rate=dropout_rate),
                            layers.Dense(layers_dimensions[1],kernel_initializer=initializer,bias_initializer='zeros'),
                            layers.BatchNormalization(),
                            layers.Activation("relu"),
                            layers.Dropout(rate=dropout_rate),
                            layers.Dense(layers_dimensions[2],kernel_initializer=initializer,bias_initializer='zeros'),
                            layers.BatchNormalization(),
                            layers.Activation("relu"),
                            layers.Dropout(rate=dropout_rate),
                            layers.Dense(layers_dimensions[3],kernel_initializer=initializer,bias_initializer='zeros'),
                            layers.BatchNormalization(),
                            layers.Activation("relu"),
                            layers.Dense(1,kernel_initializer=initializer,bias_initializer='zeros'),
                            layers.Activation("sigmoid")
])

# Define optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=1e-2)
model_nn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["binary_accuracy"])

# # Fit the model to the train set
# history_nn = model_nn.fit(X_t,y_t,validation_data=(X_v,y_v),batch_size=32,epochs=300,use_multiprocessing=True,verbose=1)
# history_nn_pd = pd.DataFrame(history_nn.history)

# Load Model Weights
model_nn.load_weights("cardiovascular_model_weights_64_0.4_fourlayers.h5")

features_mean = np.array([5.33750000e+01, 7.96511628e-01, 1.31831395e+02, 2.43881105e+02,
                         2.38372093e-01, 1.36375000e+02, 4.09883721e-01, 8.83866279e-01,
                         5.29069767e-01, 1.97674419e-01, 2.16569767e-01, 5.66860465e-02,
                         1.99127907e-01, 6.04651163e-01, 1.96220930e-01, 6.83139535e-02,
                         5.18895349e-01, 4.12790698e-01]).reshape(1,-1)
features_std = np.array([9.50575484,  0.40259267, 17.81153116, 52.72453095,  0.42608783,
                       25.89225354,  0.49181201,  1.07983022,  0.49915423,  0.3982452 ,
                        0.41190691,  0.23124173,  0.39934444,  0.48892549,  0.3971376 ,
                        0.25228388,  0.49964284,  0.4923358]).reshape(1,-1)

chest_pain_types = np.array(["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina (TA)"])
resting_ecg_types = np.array(["LVH (showing probable or definite left ventricular hypertrophy by Estes' criteria)", "Normal",
                            "ST (having ST-T wave abnormality; T wave inversions and/or ST elevation or depression of > 0.05 mV)"])
st_slope_types = np.array(["Downsloping", "Flat", "Upsloping"])

# Quantitative Data
st.write("# Curatio")
st.write("## Heart Disease Diagnosis Tool")
age = st.slider("Age (years)", min_value=0, max_value=100)
sbp = st.slider("Systolic Blood Pressure (SBP)", min_value=50, max_value=250)
chol = st.slider("Cholesterol (mg/dL)", min_value=0, max_value=400)
maxHR = st.slider("Maximum heart rate achieved)", min_value=0, max_value=250)
old_peak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", min_value=-10, max_value=10)

# Categorical Data
sex_input = st.radio("Biological Sex", ('Male', 'Female'))
chest_pain_type_input = st.radio("Chest Pain Type", ('Typical Angina (TA)', 'Atypical Angina (ATA)', 'Non-Anginal Pain (NAP)', 'Asymptomatic (ASY)'))
fasting_bs_input = st.radio("Fasting Blood Sugar", ('≥ 120 mg/dL', '< 120 mg/dL'))
resting_ecg_input = st.radio("Resting ECG Results", ('Normal', 'ST (having ST-T wave abnormality; T wave inversions and/or ST elevation or depression of > 0.05 mV)',
                                                'LVH (showing probable or definite left ventricular hypertrophy by Estes\' criteria)'))
exercise_angina_input = st.radio("Exercise-Induced Angina", ('Yes', 'No'))
st_slope_input = st.radio("ST Slope (the slope of the peak exercise ST segment)", ('Upsloping', 'Flat', 'Downsloping'))

features_list = [age, int(sex_input == "Male"), sbp, chol, int(fasting_bs_input == "≥ 120 mg/dL"), maxHR, int(exercise_angina_input == "Yes"), old_peak]
features_list += (chest_pain_types == chest_pain_type_input).astype(int).tolist()
features_list += (resting_ecg_types == resting_ecg_input).astype(int).tolist()
features_list += (st_slope_types == st_slope_input).astype(int).tolist()
features = np.array(features_list).reshape(1,-1)
# st.write(features)
features = (features - features_mean) / features_std

probability = model_nn.predict(features)

if st.button("Calculate"):
    if probability < 0.5:
        st.write("We believe that you DO NOT have heart disease.")
    else:
        st.write("We believe that you DO have heart disease.")

    st.write("The probability that you have heart disease is " + str(round(100 * float(probability),2)) + "%.")
