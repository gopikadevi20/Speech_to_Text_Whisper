from flask import Flask, render_template, request
import functools
import os
import subprocess
import torch
import torchaudio

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy-load Whisper model
@functools.lru_cache(maxsize=1)
def get_model():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return processor, model

@app.route("/", methods=["GET", "POST"])
def home():
    transcription = ""
    if request.method == "POST":
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename != "":
                original_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(original_path)

                # Convert to 16kHz mono WAV using FFmpeg
                wav_path = os.path.join(UPLOAD_FOLDER, "converted.wav")
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", original_path,
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        wav_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Load audio with torchaudio
                    speech_array, _ = torchaudio.load(wav_path)

                    # Load Whisper model
                    processor, model = get_model()
                    input_features = processor(
                        speech_array[0].numpy(),
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features

                    # Generate transcription (English)
                    generated_ids = model.generate(
                        input_features,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe")
                    )
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                except Exception as e:
                    transcription = f"❌ Error during transcription: {e}"

    return render_template("index.html", transcription=transcription)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
