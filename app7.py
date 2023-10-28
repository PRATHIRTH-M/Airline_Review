import streamlit as st
import pickle
import numpy as np

# Load the saved KNN model from the pickle file
with open('nb_model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open('nb_tfidf.pkl', 'rb') as model_file:
    tfidf = pickle.load(model_file)

# Function to predict sentiment based on user input
def predict_sentiment(review):
    tfidf_output = tfidf.transform(review)
    output = nb_model.predict(tfidf_output)
    
    return output

def other_service(seat_comfort, cabin_staff_service, food_beverages, value_for_money):
    post_list,neg_list = [],[]
    if seat_comfort > 6: seat = post_list.append(1)
    else: seat = neg_list.append(1)
    
    if cabin_staff_service > 6: cabin = post_list.append(1)
    else: cabin = neg_list.append(1)

    if food_beverages > 6: food = post_list.append(1)
    else: food = neg_list.append(1)

    if value_for_money > 6: money = post_list.append(1)
    else: money = neg_list.append(1)

    if len(post_list) > len(neg_list): return "Positive"
    elif len(post_list) == len(neg_list): return "Positive"
    else: return "Negative"
    

# Streamlit app with enhanced UI
def main():
    st.title('Airline Sentiment Analysis')

    # st.image('C:\\Users\\Prathirth\\Desktop\\PDS_MAIN\\logo.jpg', use_column_width=True)

    st.write(
        "Welcome to the Airline Sentiment Analysis App! "
        "Enter your flight experience details and share your review to get sentiment predictions."
    )


    user_review = st.text_area('Enter Your Review Here:')

    seat_comfort = st.slider('Seat Comfort', 0, 10, 5)
    cabin_staff_service = st.slider('Cabin Staff Service', 0, 10, 5)
    food_beverages = st.slider('Food & Beverages', 0, 10, 5)
    value_for_money = st.slider('Value For Money', 0, 10, 5)

    if st.button('Predict Sentiment'):
        # Make prediction
        sentiment_prediction = predict_sentiment([user_review])
        other_sentiment = other_service(seat_comfort, cabin_staff_service, food_beverages, value_for_money)

        # Display result
        st.write(f"Prediceted Sentiment from Reivew: {sentiment_prediction[0]}")
        st.write(f"Prediction from rating of other services: {other_sentiment}")
        st.success(f'Final Sentiment Prediciton: {sentiment_prediction[0] and other_sentiment}')

if __name__ == '__main__':
    main()
