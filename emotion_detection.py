import requests
import json

def emotion_detector(text_to_analyze):
    """
    Detect emotions in a given text using Watson NLP API.

    Args:
        text_to_analyze (str): The input text to analyze.

    Returns:
        dict: A dictionary containing emotion scores and the dominant emotion.
    """
    # Define the API endpoint and headers
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json"
    }

    # Define the input JSON payload
    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        # Send POST request to the Watson NLP API
        response = requests.post(url, json=input_json, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the raw response JSON and debug the API response
        response_data = response.json()
        print("Raw API Response:", json.dumps(response_data, indent=4))  # Debugging

        # Extract emotion predictions
        emotions = response_data.get('emotion_predictions', {})

        # Prepare the result with required emotions and determine the dominant emotion
        result = {
            'anger': emotions.get('anger', 0),
            'disgust': emotions.get('disgust', 0),
            'fear': emotions.get('fear', 0),
            'joy': emotions.get('joy', 0),
            'sadness': emotions.get('sadness', 0),
        }
        result['dominant_emotion'] = max(result, key=result.get) if result else None
        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from the API"}

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

