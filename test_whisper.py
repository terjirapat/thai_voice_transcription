import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- โหลดโมเดล ---
model = WhisperModel("tiny", device="cuda", compute_type="int8")

SAMPLERATE = 16000
BLOCKSIZE = 1024  # ขนาด buffer ต่อครั้ง
CHANNELS = 1

# Buffer เก็บเสียงเพื่อส่งเข้าโมเดล
audio_buffer = []

def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)
    # indata เป็น float32 [-1,1] -> แปลงเป็น int16
    audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
    audio_buffer.extend(audio_int16.tolist())

def process_buffer():
    """
    ส่ง buffer ไปทำ transcription ทุก 1-2 วินาที
    """
    global audio_buffer
    if len(audio_buffer) == 0:
        return
    # แปลงเป็น numpy array
    audio_np = np.array(audio_buffer, dtype=np.float32) / 32768.0  # กลับเป็น float32 [-1,1]
    audio_buffer = []  # ล้าง buffer

    # faster-whisper ต้อง sample rate 16kHz mono
    segments, info = model.transcribe(audio_np, beam_size=5)
    for segment in segments:
        print(segment.text)

# --- เริ่มอ่านเสียงจากไมโครโฟน ---
with sd.InputStream(channels=CHANNELS, samplerate=SAMPLERATE,
                    blocksize=BLOCKSIZE, callback=audio_callback):
    import time
    print("Recording... press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)  # ทุก 1 วินาที
            process_buffer()
    except KeyboardInterrupt:
        print("Stopped recording")
