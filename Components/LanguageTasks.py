from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API')  # Ensure this is set up in your .env file
)

def extract_times(json_string):
    """Extracts start and end times from JSON and converts them to integers."""
    try:
        data = json.loads(json_string)
        start_time = int(float(data[0]["start"]))
        end_time = int(float(data[0]["end"]))
        return start_time, end_time
    except Exception as e:
        print("Error parsing JSON:", e)
        return 0, 0

def parse_highlight_response(response_text):
    """Parses and validates JSON from the model's response."""
    try:
        # Locate JSON content by finding the brackets
        start_index = response_text.find("[")
        end_index = response_text.rfind("]") + 1
        json_str = response_text[start_index:end_index]

        # Parse JSON
        highlights = json.loads(json_str)
        
        # Check JSON structure
        if (
            isinstance(highlights, list) and
            len(highlights) == 1 and
            "start" in highlights[0] and
            "end" in highlights[0] and
            "content" in highlights[0]
        ):
            return highlights
        else:
            print("Error: Parsed JSON structure is incorrect.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

system = '''
Based on the transcription provided by the user, generate a highlight that is around 1 minute long. This highlight should be directly suitable for a short and should capture interesting parts. Please provide timestamps for the clip start and end, ensuring the highlight is a continuous part of the video.

Format response **strictly as JSON** without any extra text or explanation. Use this exact format:

[
    {
        "start": "Start time of the clip in seconds",
        "content": "Highlight text capturing main points",
        "end": "End time of the clip in seconds"
    }
]

Only one start, end, and content should be returned as JSON. Do not add any other text or explanation.
'''

def GetHighlight(transcription):
    """Fetches highlight from transcription and validates JSON output."""
    print("Getting Highlight from Transcription")
    response = client.chat.completions.create(
        model="gpt-4",  # Replace with the model you have access to
        temperature=0.7,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": transcription}
        ]
    )

    response_text = response.choices[0].message.content  # Access content directly
    highlight_data = parse_highlight_response(response_text)
    
    if highlight_data:
        start, end = extract_times(response_text)
        return start, end
    else:
        print("Error in getting highlight: invalid JSON format")
        return 0, 0

if __name__ == "__main__":
    transcription_text = "Your transcription here"
    start, end = GetHighlight(transcription_text)
    print(f"Highlight start: {start}, end: {end}")
