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
1. Blast:
- Initial symptoms are white to gray-green lesions or spots with brown borders
- Small specks originate on leaves - subsequently enlarge into spindle shaped spots(0.5 to 1.5cm length, 0.3 to 0.5cm width) with ashy center.  
- Older lesions are elliptical or spindle-shaped and whitish to gray with necrotic borders.
- Big irregular patches on leaf.
- Internodal infection also occurs at the base of the plant which causes white panicles similar to that induced by yellow stem borer or water deficit.
- Lesions on the neck are grayish brown and causes the girdling of the neck and the panicle to fall over.
- Lesions on the branches of the panicles and on the spikelet pedicels are brown to dark brown.
2. False Smut:
- Individual rice grain transformed into a mass of yellow fruiting bodies
- Growth of velvetty spores that enclose floral parts
- Infected grain has greenish smut balls with a velvetty appearance.
- The smut ball appears small at first and grows gradually up to the size of 1 cm.
- It is seen in between the hulls and encloses the floral parts.
- Only few grains in a panicle are usually infected and the rest are normal.
- As the fungi growth intensifies, the smut ball bursts and becomes orange then later yellowish-green or greenish-black in color.
- Infection usually occurs during the reproductive and ripening stages, infecting a few grains in the panicle and leaving the rest healthy.
3. Boron Deficiency:
- Reduced plant height.
- Leaf tips become white in colour and rolled.
4. Others/NA: When the provided image does not fall under any of the above three categories. Do not provide any next steps in the final response.

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

