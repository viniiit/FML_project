# Importing necessary libraries
import numpy as np
import streamlit as st
import sqlite3
from textblob import TextBlob
conn = sqlite3.connect('sentiment.db')
cursor = conn.cursor()
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Connect to the SQLite database
conn = sqlite3.connect('sentiment.db')
cursor = conn.cursor()

# Create a table for user inputs if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        sentiment TEXT
    )
''')
conn.commit()

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Dictionary mapping emotion index to emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load music data from CSV file 
df = pd.read_csv('muse_v3.csv')
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()

# Rename columns for better readability
df_sadp = df[:9000]
df_sadn = df[9000:18000]
df_fearp = df[18000:27000]
df_fearn = df[27000:36000]
df_angryp = df[36000:45000]
df_angryn = df[45000:54000]
df_neutralp = df[54000:63000]
df_neutraln = df[63000:72000]
df_happyp = df[72000:81000]
df_happyn = df[81000:]

# Split the dataframe into different emotion categories
# (e.g., df_sadp, df_sadn, df_fearp, df_fearn, etc.)

# Function to generate music recommendations based on selected emotions
def fun(emotions):
    data = pd.DataFrame()
    # Logic to select music based on the combination of emotions
    if len(emotions) == 1:
        v = emotions[0]
        t = 30

        if v == 'neutralp':
            temp = df_neutralp.sample(n=t)
            data = pd.concat([data, temp[:]])
        
        elif v == 'neutraln':
            temp = df_neutraln.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'angryp':
            temp = df_angryp.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'angryn':
            temp = df_angryn.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'fearfulp':
            temp = df_fearp.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'fearfuln':
            temp = df_fearn.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'happyp':
            temp = df_happyp.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'happyn':
            temp = df_happyn.sample(n=t)
            data = pd.concat([data, temp[:]])

        elif v == 'sadp':
            temp = df_sadp.sample(n=t)
            data = pd.concat([data, temp[:]])

        else:
            temp = df_sadn.sample(n=t)
            data = pd.concat([data, temp[:]])

    elif len(emotions) == 2:
        times = [20,10]

        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]

            if v == 'neutralp':
                temp = df_neutralp.sample(n=t)
                data = pd.concat([data, temp[:]])
        
            elif v == 'neutraln':
                temp = df_neutraln.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryp':
                temp = df_angryp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryn':
                temp = df_angryn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfulp':
                temp = df_fearp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfuln':
                temp = df_fearn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyp':
                temp = df_happyp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyn':
                temp = df_happyn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'sadp':
                temp = df_sadp.sample(n=t)
                data = pd.concat([data, temp[:]])

            else:
                temp = df_sadn.sample(n=t)
                data = pd.concat([data, temp[:]])

    elif len(emotions) == 3:
        times = [15,10,5]

        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]

            if v == 'neutralp':
                temp = df_neutralp.sample(n=t)
                data = pd.concat([data, temp[:]])
        
            elif v == 'neutraln':
                temp = df_neutraln.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryp':
                temp = df_angryp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryn':
                temp = df_angryn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfulp':
                temp = df_fearp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfuln':
                temp = df_fearn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyp':
                temp = df_happyp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyn':
                temp = df_happyn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'sadp':
                temp = df_sadp.sample(n=t)
                data = pd.concat([data, temp[:]])

            else:
                temp = df_sadn.sample(n=t)
                data = pd.concat([data, temp[:]])

    elif len(emotions) == 4:
        times = [10,9,8,3]

        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]

            if v == 'neutralp':
                temp = df_neutralp.sample(n=t)
                data = pd.concat([data, temp[:]])
        
            elif v == 'neutraln':
                temp = df_neutraln.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryp':
                temp = df_angryp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryn':
                temp = df_angryn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfulp':
                temp = df_fearp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfuln':
                temp = df_fearn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyp':
                temp = df_happyp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyn':
                temp = df_happyn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'sadp':
                temp = df_sadp.sample(n=t)
                data = pd.concat([data, temp[:]])

            else:
                temp = df_sadn.sample(n=t)
                data = pd.concat([data, temp[:]])

    else:
        times = [10,7,6,5,2]

        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]

            if v == 'neutralp':
                temp = df_neutralp.sample(n=t)
                data = pd.concat([data, temp[:]])
        
            elif v == 'neutraln':
                temp = df_neutraln.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryp':
                temp = df_angryp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'angryn':
                temp = df_angryn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfulp':
                temp = df_fearp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'fearfuln':
                temp = df_fearn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyp':
                temp = df_happyp.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'happyn':
                temp = df_happyn.sample(n=t)
                data = pd.concat([data, temp[:]])

            elif v == 'sadp':
                temp = df_sadp.sample(n=t)
                data = pd.concat([data, temp[:]])

            else:
                temp = df_sadn.sample(n=t)
                data = pd.concat([data, temp[:]])

    return data

# Function to preprocess a list and return unique elements in the order of occurrence
def pre(l):
    result = [item for items, c in Counter(l).most_common() for item in [items] * c]
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul

# Function to create a convolutional neural network (CNN) model
def create_model():
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model.h5')
    return model

# Function to insert user input into the SQLite database
def insert_user_input(user_input,sentiment):
    sentiment = 'Positive' if sentiment == 'P' else 'Negative'
    cursor.execute('INSERT INTO user_inputs (user_input,sentiment) VALUES (?,?)', (user_input,sentiment,))
    conn.commit()

# Function to retrieve user inputs from the database
def get_user_inputs():
    cursor.execute('SELECT user_input,sentiment FROM user_inputs')
    data = cursor.fetchall()
    return data

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity >= 0:
        return 'P'
    else:
        return 'N'

# Main function to run the Streamlit app
def main():
    model = create_model()
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    
    st.markdown("<h2 style='text-align: center; color: grey;'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)
 
    col1,col2,col3 = st.columns(3)

    with col1:
        c1,c2 = st.columns(2)
        c1.subheader("Text")
        c2.subheader("Sentiment")
        user_inputs = get_user_inputs()
        user_inputs.reverse()
        for input_text,sentiment in user_inputs:
            c1.write(f"- {input_text[:15]}")
            c2.write(f"- {sentiment}")
    
    submit = False
    user_input=""
    with col2:
        user_input = st.text_area("Enter Text:", height=500)
        sentiment = sentiment_analysis(user_input)
        c1,c2,c3 = st.columns(3)
        submit = col2.button("Submit")
        if submit and user_input != "":
            insert_user_input(user_input,sentiment)

    new_emotions = []
    with col3:
        if submit and user_input != "":
            count = 0
            emotions = []
            while True:
                emotions.clear()
                ret, frame = cap.read()
                if not ret:
                    break
                face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                count = count + 1

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))
                    emotions.append(emotion_dict[max_index])
                    cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

                if cv2.waitKey(1) & 0xFF == ord('x'):
                    break

                if count >= 20:
                    break

            cap.release()
            cv2.destroyAllWindows()

            emotions = pre(emotions)

            for i in emotions:
                new_emotions.append(i.lower()+sentiment)

        elif submit and user_input == "":
            st.error("Please enter text to get recommendations")

    new_df = fun(new_emotions)

    col3.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>", unsafe_allow_html=True)
    col3.write("---------------------------------------------------------------------------------------------------------------------")

    try:
        for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
            col3.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(l,i+1,n),unsafe_allow_html=True)
            col3.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a), unsafe_allow_html=True)
            col3.write("---------------------------------------------------------------------------------------------------------------------")
    except:
        pass

# Run the Streamlit app if the script is executed directly
if __name__ == '__main__':
    main()
