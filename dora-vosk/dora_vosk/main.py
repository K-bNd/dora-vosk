"""Real-time speech recognition using Vosk with direct microphone input."""

import pyarrow as pa
from dora import Node

import re
import json
import os

from vosk import Model, KaldiRecognizer

BAD_SENTENCES = [
    "",
    " so",
    " So.",
    " so so",
    " What?",
    " Here we go.",
    " my",
    " All right. Thank you.",
    " That's what we're doing.",
    " That's what I wanted to do.",
    " I'll be back.",
    " And we'll see you next time.",
    "You",
    "You ",
    " You",
    " you",
    "!",
    "THANK YOU",
    " Thank you.",
    " The",
]

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '16000'))
BLOCKSIZE = int(os.getenv('BLOCKSIZE', '8000'))  # 0.5 seconds at 16000 Hz


def remove_text_noise(text: str, text_noise="") -> str:
    """Remove noise from text."""
    if not text_noise.strip():
        return text

    def normalize(s):
        s = re.sub(r"-", " ", s)
        return re.sub(r"[^\w\s]", "", s).lower()

    normalized_text = normalize(text)
    normalized_noise = normalize(text_noise)

    text_words = normalized_text.split()
    noise_words = normalized_noise.split()

    cleaned_words = text_words[:]
    for noise_word in noise_words:
        if noise_word in cleaned_words:
            cleaned_words.remove(noise_word)

    return " ".join(cleaned_words)


def process_text(text: str):
    """Process recognized text and send output."""
    # Check bad sentences
    if text in BAD_SENTENCES:
        return

    # Skip empty text
    if text.strip() == "" or text.strip() == ".":
        return

    return text


def main():
    """Run Vosk Model for speech recognition."""
    node = Node()
    print("Loading Vosk model...")
    model = Model(lang="en-us")
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    print("Vosk model loaded")

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "audio":

                data = event['value'].to_numpy()

                # Feed to Vosk recognizer
                if recognizer.AcceptWaveform(data.tobytes()):
                    # Final result
                    result_str = recognizer.Result()
                    result = json.loads(result_str)

                    text = result.get('text', '').strip()

                    if text:
                        print(f"\nFinal: {text}")
                        text = process_text(text)
                        node.send_output(
                            output_id="text", data=pa.array([text]), metadata={},
                        )
                else:
                    print("Partial....")
                    # Partial result
                    partial_str = recognizer.PartialResult()
                    partial = json.loads(partial_str)

                    if partial.get('partial'):
                        print(f"Partial: {partial['partial']}", end='\r')
                    node.send_output(
                        output_id="text", data=pa.array([partial.get('partial')]), metadata={},
                    )


if __name__ == "__main__":
    main()
