import streamlit as st
import tempfile
import os

from pydub import AudioSegment
import numpy as np
from whisperspeech.pipeline import Pipeline

pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')

def voice_generation(text, speaker):
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.close()  # Close the file so that whisperspeech can use it
        audio_tensor = pipe.generate(text, lang='en', speaker=speaker)
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)

        if len(audio_np.shape) == 1:
            audio_np = np.expand_dims(audio_np, axis=0)
        else:
            audio_np = audio_np.T

        try:
            audio_segment = AudioSegment(
                audio_np.tobytes(),
                frame_rate=24000,
                sample_width=2,
                channels=1
            )
            audio_segment.export('output_audio.wav', format='wav')
            print("Audio file generated: output_audio.wav")
        except Exception as e:
            print(f"Error writing audio file: {e}")

        return "output_audio.wav"

def main():
    st.title("Voice Generation System")

    # User input prompts
    user_input = st.text_input("Enter text to convert to Voice:")
    speaker = st.text_input("Enter speaker URL (optional):")

    # Button to trigger text-to-speech conversion
    if st.button("Convert to Speech"):
        if user_input:
            audio_file_path = voice_generation(user_input, speaker)
            audio_bytes = open(audio_file_path, 'rb').read()
            st.audio(audio_bytes, format='audio/wav')
            os.remove(audio_file_path)  # Remove the temporary audio file after playing
            st.success("Speech generated successfully!")
        else:
            st.warning("Please enter some text!")

if __name__ == "__main__":
    main()
