
#Imported all the necessary files

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from scipy import stats
from gtts import gTTS
import numpy as np



#Holistic model
mp_holi = mp.solutions.holistic

#Drawing utilities
mp_draw = mp.solutions.drawing_utils


def mediapipe_detect(image, model):

    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image is no longer writeable
    image.flags.writeable = False

    # Make prediction
    results = model.process(image)

    # Image is now writeable
    image.flags.writeable = True

    # COLOR COVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


#To improve visual effect in landmarks connections
def landmarks(image, results):
    #
    # Modified for face connections
    mp_draw.draw_landmarks(image, results.face_landmarks, mp_holi.FACEMESH_TESSELATION,
                             mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                              )
    # Modified for pose connections
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holi.POSE_CONNECTIONS,
                             mp_draw.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_draw.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Modified for hand connections
    mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holi.HAND_CONNECTIONS,
                             mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_draw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Modified for hand connections
    mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holi.HAND_CONNECTIONS,
                             mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


#Check if are able to detect all the landmarks using mediapipe_detect function

#For webcam the value is 0
cap = cv2.VideoCapture(0)

# First we will set mediapipe model
with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        #Read each frame
        ret, frame = cap.read()

        #  getting landmarks  fro rach frame using detections
        image, results = mediapipe_detect(frame, holistic)
        print(results)

        # Improving landmarks
        landmarks(image, results)

        # Display the image
        cv2.imshow('OpenCV Feed', image)

        # Press "s" for break from loop
        if cv2.waitKey(10) & 0xFF == ord('s'):
            break

    #Once done release and: destroy all windows
    cap.release()
    cv2.destroyAllWindows()


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#Extracting values and storing it as array
def extract_values(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh, rh])


# Path where each class of images will be stored
DATA_PATH = os.path.join('/Users/varun/Documents/Data Science/AML/Models/Data1')

# Final classes which we will be detecting
actions = np.array(['hello','thanks','yes','no'])

# Total thirty five vedios are created for each class with different postions
no_sequences = 35

# Each vedios contains thirty frames
sequence_length = 30



# #Making each folder for each class to store vedios
for action in actions:
    for sequence in range(1,no_sequences+1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action,str(sequence)))
        except:
            pass

# #This will be the dataset for our project
# #Capturing vedio starts
cap = cv2.VideoCapture(0)

with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Loop through different actions
    for action in actions:
        # Loop through videos
        for sequence in range(1,no_sequences):

            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detect(frame, holistic)

                landmarks(image, results)

                # Reading 30 frames
                if frame_num == 0:
                    cv2.putText(image, ' COLLECTION FRAMES', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(3000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # Keyvalue points are stored in below path
                keypoints = extract_values(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break with "q"
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


#Numbering for each class
label_map={label:num for num ,label in enumerate(actions)}


#Lableiing for each vedios
sequences, labels = [], []
for action in actions:
    for sequence in range(1,no_sequences):
        window = []

        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

#Variables
X = np.array(sequences)

#Target
Y = to_categorical(labels).astype(int)

#Train and test split with 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

#Keeping track of training
log_dir = os.path.join('/Users/varun/Documents/Data Science/AML/Models/Fifth/Logs')

#Training accuracy and loss can be found with tensorboard.
#Also we can see how model structure is created using tensorboard
tb_callback = TensorBoard(log_dir=log_dir)

# #Model Training
#
# #Input is 1662
# # 468*3 + 33*4 + 21*3  +   21*3        = 1662
# # face    pose  lefthand  righthand
#
# # Five LSTM layers
# # Three Dense layers
#
models = Sequential()
models.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
models.add(LSTM(128, return_sequences=True, activation='relu'))
models.add(LSTM(256, return_sequences=True, activation='relu'))
models.add(LSTM(128, return_sequences=True, activation='relu'))
models.add(LSTM(64, return_sequences=False, activation='relu'))
models.add(Dense(64, activation='relu'))
models.add(Dense(32, activation='relu'))
models.add(Dense(actions.shape[0], activation='softmax'))

#categorical_crossentropy for multiclass
models.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#Training of model starts
models.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])

#Saving the model
models.save('/Users/varun/Documents/Data Science/AML/Models/Fifth')

#Loading the model again
new_model=tensorflow.keras.models.load_model('/Users/varun/Documents/Data Science/AML/Models/Fifth')

#
#Predict for X_train and X_test
ytrain_pred=new_model.predict(X_train)
ytest_pred=new_model.predict(X_test)

#Actual class
ytrain_true=np.argmax(y_train, axis=1).tolist()
ytest_true = np.argmax(y_test, axis=1).tolist()

#Predicted class
ytrain_hat = np.argmax(ytrain_pred, axis=1).tolist()
ytest_hat = np.argmax(ytest_pred, axis=1).tolist()

#Confusion matrix
print('Confusion matrix of training data',multilabel_confusion_matrix(ytrain_true, ytrain_hat))
print('Confusion matrix of testing data',multilabel_confusion_matrix(ytest_true, ytest_hat))


# #LIVE FEED

#Four colors for four classes
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),(200, 120, 20)]


#Creating class detection on left side of screen
def dimensions(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

# Detection variables
sequence = []
sentence = []
predictions = []

#Threshold set to 80%
threshold = 0.80

#Live feed starts, detection is done, class has been predicted then it will be showing on screen
cap = cv2.VideoCapture(0)

with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mediapipe_detect(frame, holistic)
        print(results)

        landmarks(image, results)

        keypoints = extract_values(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Append only when class has been changed
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            #Display only 5 class with history at once each time
            if len(sentence) > 5:
                sentence = sentence[-5:]

            #Probabilities
            image = dimensions(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break with key "q"
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#
























