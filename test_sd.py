import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024  # samples per callback

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_block = indata[:, 0]  # mono
    # ส่ง audio_block ไปยังโมเดล ASR ได้ทันที
    print("Audio block:", audio_block[:5])

print("🎤 Start capturing (Ctrl+C to stop)")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    while True:
        sd.sleep(100)
