"""Real-time speech recognition using Vosk with direct microphone input."""
import re
import time
import json
import queue
import traceback
import os

import numpy as np
import pyarrow as pa
import sounddevice as sd
from dora import DoraStatus
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


class Operator:
    def __init__(self):
        self.text_noise = ""
        self.noise_timestamp = time.time()
        self.stream = None
        self.stream_started = False
        self.running = True

        # Audio queue - callback just adds to queue
        self.audio_queue = queue.Queue()

        # Initialize Vosk
        print("Loading Vosk model...")
        self.model = Model(lang="en-us")
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        print("Vosk model loaded")

    def on_event(self, dora_event, send_output) -> DoraStatus:
        # Start the audio stream on first event
        if not self.stream_started:
            self.start_stream()
            self.stream_started = True

        if dora_event["type"] == "INPUT":
            return self.on_input(dora_event, send_output)
        elif dora_event["type"] == "STOP":
            self.running = False
            self.stop_stream()
            return DoraStatus.STOP

        return DoraStatus.CONTINUE

    def on_input(self, dora_input, send_output):
        # Handle text_noise input
        if "text_noise" in dora_input["id"]:
            self.text_noise = dora_input["value"][0].as_py()
            self.text_noise = (
                self.text_noise.replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
            )
            self.noise_timestamp = time.time()
            print(f"Updated text noise: {self.text_noise}")

        # Process all queued audio
        processed_any = False
        while not self.audio_queue.empty():
            try:
                data = self.audio_queue.get_nowait()

                # Feed to Vosk recognizer
                if self.recognizer.AcceptWaveform(data.tobytes()):
                    # Final result
                    result_str = self.recognizer.Result()
                    result = json.loads(result_str)

                    text = result.get('text', '').strip()

                    if text:
                        self.process_text(text, send_output)
                        processed_any = True
                # else:
                #     # Partial result
                #     partial_str = self.recognizer.PartialResult()
                #     partial = json.loads(partial_str)

                    # if partial.get('partial'):
                    #     print(f"Partial: {partial['partial']}", end='\r')

            except queue.Empty:
                break
            except Exception as e:
                print(f"ERROR processing audio: {e}")
                traceback.print_exc()

        if processed_any:
            print()  # Newline after partial results

        return DoraStatus.CONTINUE

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback - just queues audio for processing."""
        if status:
            print(f"Audio status: {status}")

        # Just queue the audio data as numpy array
        self.audio_queue.put(np.array(indata))

    def process_text(self, text, send_output):
        """Process recognized text and send output."""
        try:
            # Check bad sentences
            if text in BAD_SENTENCES:
                return

            # Remove noise filter after some time
            if time.time() - self.noise_timestamp > (len(self.text_noise.split()) / 2):
                self.text_noise = ""

            # Remove text noise
            if self.text_noise:
                text = remove_text_noise(text, self.text_noise)

            # Skip empty text
            if text.strip() == "" or text.strip() == ".":
                return

            # Check if text is Chinese
            is_chinese = re.findall(r"[\u4e00-\u9fff]+", text)

            if is_chinese or not text.endswith("..."):

                send_output(
                    "text",
                    pa.array([text]),
                    {"primitive": "text"},
                )
                send_output(
                    "speech_started",
                    pa.array([text]),
                )
           
        except Exception as e:
            print(f"ERROR processing text: {e}")
            traceback.print_exc()

    def start_stream(self):
        """Start the audio input stream."""
        print("Starting audio stream...")

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            )
            self.stream.start()
            print("Audio stream started")
        except Exception as e:
            print(f"ERROR starting stream: {e}")
            traceback.print_exc()

    def stop_stream(self):
        """Stop the audio input stream."""
        if self.stream is not None:
            print("Stopping audio stream...")
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def __del__(self):
        """Cleanup when operator is destroyed."""
        self.running = False
        self.stop_stream()