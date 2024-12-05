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

        # Parse the response JSON
        response_data = response.json()
        emotion_data = response_data.get('emotionPredictions', [])[0]['emotion']

        # Prepare the result with required emotions
        result = {
            'anger': emotion_data.get('anger', 0),
            'disgust': emotion_data.get('disgust', 0),
            'fear': emotion_data.get('fear', 0),
            'joy': emotion_data.get('joy', 0),
            'sadness': emotion_data.get('sadness', 0),
        }

        # Determine the dominant emotion
        result['dominant_emotion'] = max(result, key=result.get)
        return result

    except (IndexError, KeyError):
        return {"error": "Unexpected API response structure"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from the API"}

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
