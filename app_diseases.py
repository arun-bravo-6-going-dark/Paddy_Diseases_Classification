import streamlit as st
import os
import base64
import requests
import json
from PIL import Image
import io
import re
import pandas as pd

# Function to encode the image
def encode_image(image):
    try:
        return base64.b64encode(image.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

# Function to classify an image
def classify_image(base64_image, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    sys_prompt = """Analyze the provided images of rice plants and identify the presence of the following diseases based on the visual symptoms described:
1. Leaf Scald:
   - Long, irregular, water-soaked streaks or lesions that dry into a whitish-gray, "scorched" appearance.
   - Lesions start at leaf tips or edges and progress downward.
   - Leaves appear wilted or drought-affected in severe cases.

2. False Smut:
   - Grain transformed into greenish-yellow smut balls with a velvetty appearance.
   - Smut balls grow up to 1 cm, burst into orange or greenish-black spore masses.
   - Only a few grains in a panicle are affected, others remain normal.

3. Boron Deficiency:
   - White discoloration and rolling of leaf tips.
   - Reduced plant height and stunted growth.

4. Others/NA:
   - When the provided image does not clearly match the above conditions.

You should also provide the farmer with next steps to deal with this Paddy disease in two sentences.

Use these visual symptoms to classify and identify the specific disease affecting the rice plants in the images provided. The final response should be in the following format:
```
{
"disease_name": "<disease name>",
"confidence_score": "<confidence score>",
"next_steps": "<next steps to deal with this disease>",
}
```"""

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": sys_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the disease in the rice plant."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
        "top_p": 0.5,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return None
    
# Streamlit UI
# Set wide mode as default
st.set_page_config(layout="wide")
st.title("🌾🔍 Paddy Diseases Classification 🌾🔍")

st.subheader("Classification for Paddy Diseases: Blast, False Smut, and Boron Deficiency Only ", divider="gray")

api_key = st.secrets["openai_api_key"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "gif", "png"])


if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption='📸 Uploaded Image', use_column_width=False, width=250)
        if st.button("Classify Image", type="primary"):
            base64_image = encode_image(uploaded_file)
            if base64_image:
                response_json = classify_image(base64_image, api_key)
                if response_json:
                    try:
                        response_content = response_json['choices'][0]['message']['content']
                        
                        # Define the regex pattern to extract content including the curly braces
                        pattern = r'\{.*?\}'

                        # Use re.search to find the first match of the pattern
                        match = re.search(pattern, response_content, re.DOTALL)

                        if match:
                            # Extract the content including the braces
                            content = match.group(0)
                        else:
                            st.error("Unexpected response format. Please try again.")
                            content = "{}"  # Assigning empty json structure to avoid json.loads error

                        response_data = json.loads(content)
                        print(response_data)
                        
                        with col2:
                            st.subheader("Classification Result")
                            df = pd.DataFrame({
                                "🦠 Disease Name 🌱": [response_data['disease_name']],
                                "🔍 Confidence Score 🔍": [response_data['confidence_score']],
                                "🛠️ Next Steps 🛠️": [response_data['next_steps']],
                            })
                            st.markdown(df.to_html(index=False), unsafe_allow_html=True)
                            
                    except (KeyError, json.JSONDecodeError) as e:
                        st.error(f"Unexpected response format: {response_json}")

