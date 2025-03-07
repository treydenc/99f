from flask import Flask, request, jsonify
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import openai
import os
from playsound import playsound
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import pygame

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# ✅ Allow requests from frontend (localhost:3000)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# OpenAI API Key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get absolute path to elephant.mp3
ELEPHANT_SOUND = os.path.join(os.path.dirname(__file__), "elephant.mp3")

# Initialize pygame for sound playback
pygame.init()
pygame.mixer.init()

# Check if API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OpenAI API Key is missing! Set it in the .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)  # New OpenAI client format

# Audio settings
SAMPLE_RATE = 44100  # Standard sampling rate
DURATION = 10  # 10 seconds
AUDIO_FILENAME = "audio.wav"
ELEPHANT_SOUND = "elephant.mp3"  # Place an actual elephant sound file in the project

def record_audio():
    print("🔴 Recording started...")
    recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    write(AUDIO_FILENAME, SAMPLE_RATE, recording)
    print(f"✅ Recording complete. Audio saved as {AUDIO_FILENAME}")

def transcribe_audio():
    print("🎤 Sending audio to Whisper API for transcription...")
    try:
        with open(AUDIO_FILENAME, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcribed_text = response.text
        print(f"📝 Transcribed Text: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return ""

def rate_sexuality(text):
    print("🤖 Sending text to OpenAI for rating...")
    prompt = f"You are an elephant at a party. Rate how arousing the following flirting attempt feels to you on a scale from 1-10. Respond with just the number. Dont be too tough. You should be aroused when people say sexual things to you. \n\nText: \"{text}\""
    try:
        response = client.chat.completions.create( 
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        rating = int(response.choices[0].message.content.strip())
        print(f"📊 Sexuality Rating: {rating}")
        return rating
    except Exception as e:
        print(f"❌ Error during rating: {e}")
        return 0  # Default to 0 in case of an error

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    record_audio()
    transcribed_text = transcribe_audio()

    if not transcribed_text:
        return jsonify({"message": "Error transcribing audio.", "rating": 0})

    rating = rate_sexuality(transcribed_text)

    if rating >= 6:
        print("🔊 Playing the elephant sound!")

        try:
            pygame.mixer.music.load(ELEPHANT_SOUND)  
            pygame.mixer.music.play()

            return jsonify({"message": "The elephant has spoken! 🚨", "rating": rating})
        except Exception as e:
            print(f"❌ Error playing sound: {e}")
            return jsonify({"error": "Failed to play elephant sound"}), 500

    elif rating < 4:
        ANGRY_SOUND = os.path.join(os.path.dirname(__file__), "angry.mp3")
        print("😡 Playing the angry sound!")

        try:
            pygame.mixer.music.load(ANGRY_SOUND)  
            pygame.mixer.music.play()

            return jsonify({"message": "😡 The elephant is NOT impressed! Try harder!", "rating": rating})
        except Exception as e:
            print(f"❌ Error playing angry sound: {e}")
            return jsonify({"error": "Failed to play angry sound"}), 500

    else:
        witty_responses = [
            "Not even a trunk twitch. Try again. 🐘",
            "The elephant just re-downloaded a dating app out of sheer boredom.",
            "Still dry as the Sahara. ",
            "It takes more than peanuts 🥜 to get this party started.",
            "This is rated E for Elephant-friendly. Try harder."
        ]
        witty_message = np.random.choice(witty_responses)
        print(f"🐘 Response: {witty_message}")
        return jsonify({"message": witty_message, "rating": rating})

@app.route("/debug-trigger", methods=["POST"])
def debug_trigger():
    print("🐘 DEBUG: Manually triggering the elephant sound!")

    try:
        pygame.mixer.music.load(ELEPHANT_SOUND)
        pygame.mixer.music.play()
        return jsonify({"message": "🔊 Debug Mode: The elephant has spoken! 🚨 (Rating: 10)"}), 200
    except Exception as e:
        print(f"❌ Error playing sound: {e}")
        return jsonify({"error": "Failed to play elephant sound"}), 500


if __name__ == "__main__":
    print("🚀 Server is running on http://127.0.0.1:5000")
    app.run(debug=True)
