import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import re
from easyocr import Reader

#train_data = pd.read_csv("train.csv")
#train_data = train_data.iloc[14000:16000]
#train_data =train_data.tail(1000)
test_data =pd.read_csv("test_file_4.csv")
#test_data = test_data.head(100)
#test_data = test_data.iloc[400:500]
# Initialize EasyOCR reader
reader = Reader(['en'])


unit_mapping = {
    'g': 'gram',
    'grams': 'gram',
    'gram': 'gram',
    'kg': 'kilogram',
    'k9':'kilogram',
    'kilogram': 'kilogram',
    'kilograms': 'kilogram',
    'lb': 'pound',
    'lbs': 'pound',
    'pound': 'pound',
    'pounds': 'pound',
    'oz': 'ounce',
    'OZ':'ounce',
    '0z':'ounce',
    'ounce': 'ounce',
    'ounces': 'ounce',
    'cm': 'centimetre',
    'centimetre': 'centimetre',
    'cms': 'centimetre',
    'inch': 'inch',
    'inches': 'inch',
    'ii':'inches',
    'in':'inch',
    '"':'inch',
    'foot': 'foot',
    'feet': 'foot',
    'm':'metre',
    'millimetre': 'millimetre',
    'mm': 'millimetre',
    'millimetres': 'millimetre',
    'yard': 'yard',
    'yards': 'yard',
    'volt': 'volt',
    'volts': 'volt',
    'u':'volt',
    'kilovolt': 'kilovolt',
    'millivolt': 'millivolt',
    'v': 'volt',
    'kV':'kilovolt',
    'V':'volt',
    'W':'watt',
    'w':'watt',
    'wt':'watt',
    'watt': 'watt',
    'watts': 'watt',
    'kilowatt': 'kilowatt',
    'kw': 'kilowatt',
    'kW': 'kilowatt',
    'litre': 'litre',
    'litres': 'litre',
    'liter': 'litre',
    'liters': 'litre',
    'millilitre': 'millilitre',
    'ml': 'millilitre',
    'millilitres': 'millilitre',
    'gallon': 'gallon',
    'gallons': 'gallon',
    'pint': 'pint',
    'pints': 'pint',
    'quart': 'quart',
    'quarts': 'quart'
}


    

def extract_text_from_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    results = reader.readtext(img)
    extracted_text = ' '.join([result[1] for result in results])
    return extracted_text

def extract_value_based_on_entity(extracted_text, entity_name):
    patterns = {
      'item_weight': r'(\d+\.?\d*)\s?(g|grams|gram|kg|k9|kilogram|kilograms|lb|lbs|pound|pounds|oz|OZ|0z|ounce|ounces)?',
      'width': r'(\d+\.?\d*)\s?(centimetre|centimeters|cm|cms|foot|feet|inch|inches|"|ii|in|metre|metres|meter|meters|millimetre|millimetres|mm|yard|yards)?',
      'depth': r'(\d+\.?\d*)\s?(centimetre|centimeters|cm|cms|foot|feet|inch|inches|"|ii|in|metre|metres|meter|meters|millimetre|millimetres|mm|yard|yards)?',
      'height': r'(\d+\.?\d*)\s?(centimetre|centimeters|cm|cms|foot|feet|inch|inches|"|in|metre|metres|meter|meters|millimetre|millimetres|mm|yard|yards)?',
      'voltage': r'(\d+)\s?(volt|volts|kilovolt|kilovolts|millivolt|millivolts|v|u|V|kV)?',
      'wattage': r'(\d+)\s?(wt|w|W|watt|watts|kilowatt|kw|kW|kilowatts)?',
      'item_volume': r'(\d+\.?\d*)\s?(litre|litres|liter|liters|millilitre|millilitres|ml|gallon|gallons|pint|pints|quart|quarts)?'
    }

    pattern = patterns.get(entity_name)
    if not pattern:
        return None


    matches = re.findall(pattern, extracted_text, re.IGNORECASE)

   
    print(f"Matches found: {matches}")

 
    values = []
    for value, unit in matches:
     
        normalized_unit = unit_mapping.get(unit.lower(), unit)
        if normalized_unit in unit_mapping.values():
            values.append((float(value), normalized_unit))
    
    if values:
        if entity_name in ['height', 'width', 'depth']:
  
            if entity_name == 'height':

                value, unit = max(values, key=lambda x: x[0])
            elif entity_name == 'width':
                
                sorted_values = sorted(values, key=lambda x: x[0], reverse=True)
                if len(sorted_values) >= 2:
                    value, unit = sorted_values[1]
                else:
                    value, unit = sorted_values[0]  
            elif entity_name == 'depth':
       
                value, unit = min(values, key=lambda x: x[0])
        else:

            value, unit = max(values, key=lambda x: x[0])


        return f"{value} {unit_mapping.get(unit.lower(), unit)}".strip() if unit else str(value)

    return None

# Load the dataset
# Replace 'path_to_test.csv' with the actual path to the test dataset.
# test_data = pd.read_csv("path_to_test.csv")
# train_data = pd.read_csv("train.csv")

# train_data = train_data.tail(200)
# test_data = pd.read_csv("/content/drive/MyDrive/Data/Amazon_ML_Challenge/test.csv")

# Initialize a new DataFrame for predictions
predictions = pd.DataFrame(columns=['index', 'prediction'])
predictions['prediction'] = None

for i, row in test_data.iterrows():
    print(f"Processing row {i+1}/{len(test_data)}")

    image_url = row['image_link']
    entity_name = row['entity_name']

   
    extracted_text = extract_text_from_image(image_url)
    print(f"Extracted text: {extracted_text}")

    
    predicted_value = extract_value_based_on_entity(extracted_text, entity_name)
    print(f"Entity: {entity_name}")
    print(f"Extracted value: {predicted_value}")

    
    new_row = pd.DataFrame({'index': [row['index']], 'prediction': [predicted_value]})

    
    predictions = pd.concat([predictions, new_row], ignore_index=True)

predictions.to_csv('predictions_new_4.csv', index=False)




