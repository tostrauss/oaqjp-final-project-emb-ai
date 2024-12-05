import requests

def emotion_detector(text_to_analyze):
    # Define the API endpoint and headers
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json"
    }

    # Define the input JSON
    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        # Send POST request to the Watson NLP API
        response = requests.post(url, json=input_json, headers=headers)
        response.raise_for_status()

        # Extract and return the "text" attribute from the response
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
