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
- Elaborate Symptoms:
  - Initial symptoms appear as long, irregular, water-soaked streaks or lesions on the leaf blade.
  - Lesions are light brown to grayish with yellowish margins, often starting at the tip or edges of the leaves and progressing downward.
  - As the disease advances, the lesions dry out, turning whitish-gray or light tan, giving the leaf a "scorched" or "burnt" appearance.
  - In severe cases, individual lesions merge to cover large portions of the leaf blade, causing partial or total drying.
  - The affected leaves wilt, droop, and appear as if suffering from water stress or drought.
  - The disease thrives in warm, humid climates, particularly where poor drainage or prolonged wetness is present.
  - Severe infections can cause significant damage to the crop by reducing photosynthesis.

- Similar Symptoms to Differentiate:
  - Brown Spot: Smaller, circular lesions with distinct brown centers and yellow halos. Lesions do not give the leaf a scorched appearance.
  - Bacterial Leaf Blight (BLB): Yellow streaks originating from the leaf margins, spreading downward with a wet, water-soaked appearance initially.
  - Blast Disease: Spindle-shaped lesions with gray centers and dark brown borders, often smaller and not irregular in shape.

2. False Smut:
- Elaborate Symptoms:
  - Individual rice grains are transformed into masses of yellowish or greenish smut balls.
  - Velvetty fungal spores grow on the surface of the infected grains and often enclose the floral parts.
  - Early infections produce small smut balls, but these grow gradually, reaching up to 1 cm in size.
  - Smut balls first appear greenish-yellow and later change to orange, yellowish-green, or even greenish-black as they mature.
  - The disease primarily affects a few grains within a panicle, leaving the remaining grains unaffected.
  - As fungal growth intensifies, the smut balls burst, releasing spores that can spread the disease further.
  - Infection occurs during the reproductive and ripening stages of the crop, significantly reducing grain quality.
  - False Smut does not cause complete panicle sterility but impacts marketability and quality of the grain.

- Similar Symptoms to Differentiate:
  - Kernel Smut: A black powdery mass develops on infected grains, not velvetty smut balls. The disease affects multiple grains more uniformly.
  - Grain Discoloration (Bacterial or Fungal): Infected grains turn brown or black but do not develop ball-like structures.
  - Blast (Neck Rot Stage): Causes neck girdling, leading to a complete failure of grain development and white, empty panicles.

3. Boron Deficiency:
- Elaborate Symptoms:
  - Affected plants exhibit stunted growth, resulting in reduced plant height.
  - Leaf tip discoloration is one of the first symptoms, where tips turn white and appear dried out.
  - Leaves may also become rolled, distorted, or show signs of brittleness.
  - Boron deficiency affects the overall structural development of the plant, particularly in young, growing tissues.
  - If left unaddressed, symptoms can progress to reduced tillering, poor grain filling, and low yields.
  - Boron deficiency is often associated with sandy soils, acidic soils, or areas where excessive irrigation leaches out nutrients.

- Similar Symptoms to Differentiate:
  - Nitrogen Deficiency: Causes pale green or yellowing of the entire leaf, not just the tips, along with poor growth.
  - Potassium Deficiency: Causes yellowing and browning along the edges (margins) of the leaves rather than at the tips.
  - Bacterial Leaf Blight: Leads to yellow streaks progressing to brown lesions along the margins and downward drying of the leaves, not specific white discoloration at the tips.

4. Others/NA:
- When the provided image does not fall under any of the above three categories.  
- No next steps will be provided in this case.

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

