import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image
import random
import pandas as pd
import os

# Initialize FER detector
emotion_detector = FER()

# Food recommendations based on emotions
food_recommendations = {
    "happy": {
        "Appetizer": ["Onion Rings", "Popcorn Chicken"],
        "Main Course": ["Beef Wellington", "Chicken Satay", "Beef Burger"],
        "Dessert": ["Apple Pie", "Sundae"],
        "Drink": ["Margarita", "Grape Mocktail", "Mojito", "Virgin Mojito"]
    },
    "sad": {
        "Appetizer": ["Cheese Platter", "Ceaser Salad"],
        "Main Course": ["Ratatouille", "Pesto Pasta"],
        "Dessert": ["Chocolate Cake", "Carrot Cake"],
        "Drink": ["Grape Mocktail", "Virgin Mojito"]
    },
    "angry": {
        "Appetizer": ["Truffle Fries"],
        "Main Course": ["Margherita Pizza", "Ha Chong Gai"],
        "Dessert": ["Waffle Ice-cream"],
        "Drink": ["Margarita"]
    },
    "neutral": {
        "Appetizer": ["Avocado Toast"],
        "Main Course": ["Chicken Satay", "Beef&Chips"],
        "Dessert": ["Rasberry Ice-cream"],
        "Drink": ["Grape Mocktail", "Virgin Mojito"]
    }
}

# Function to recommend food based on emotion
def recommend_food(emotion):
    recommendations = food_recommendations.get(emotion, {})
    if not recommendations:
        return "No recommendation available."
    
    recommendation_text = f"Food recommendations for {emotion}:\n"
    recommendation_dict = {}
    for category, items in recommendations.items():
        selected_item = random.choice(items)
        recommendation_text += f"{category}: {selected_item}\n"
        recommendation_dict[category] = selected_item
    
    return recommendation_text, recommendation_dict

# Function to save data to CSV
def save_to_csv(emotion, food_recommendation, ordered=False):
    # Check if file exists
    file_exists = os.path.isfile("emotion_data.csv")
    
    # Create a DataFrame for the new data
    data = {
        "Emotion": [emotion],
        "Recommendation": [str(food_recommendation)],
        "Ordered": [ordered]
    }
    df = pd.DataFrame(data)

    # Append data to CSV, or create file if it doesn’t exist
    df.to_csv("emotion_data.csv", mode='a', index=False, header=not file_exists)

# Streamlit App
st.title("Face Emotion Recognition and Food Recommendation App")

# Option to capture an image from the camera
camera_input = st.camera_input("Capture a photo...")

if camera_input is not None:
    # Load image from the camera input and convert to RGB
    img = Image.open(camera_input).convert("RGB")
    img_array = np.array(img)

    # Check that the image has been loaded successfully
    if img_array is None:
        st.error("Error loading image. Please try again.")
    else:
        st.image(img, caption="Captured Image", use_column_width=True)

        # Detect emotions using FER
        emotions = emotion_detector.detect_emotions(img_array)

        # Check that emotions were detected
        if not emotions:
            st.warning("No faces or emotions detected. Try capturing a clearer image.")
        else:
            # Process each detected face
            for face in emotions:
                # Get bounding box
                (x, y, w, h) = face["box"]

                # Extract the dominant emotion
                emotion = max(face["emotions"], key=face["emotions"].get)

                # Draw bounding box and label
                img_array = cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_array, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display food recommendation for detected emotion
                st.subheader(f"Detected Emotion: {emotion}")
                food_recommendation_text, food_recommendation_dict = recommend_food(emotion)
                st.text(food_recommendation_text)

                # Save detected emotion and recommendation to CSV
                save_to_csv(emotion, food_recommendation_dict)

                # Add an order button for the recommended food
                if st.button(f"Order {emotion.capitalize()} Recommendation"):
                    st.success("Order placed successfully!")
                    save_to_csv(emotion, food_recommendation_dict, ordered=True)

            # Display image with detections
            st.image(img_array, caption="Detected Faces and Emotions", use_column_width=True)