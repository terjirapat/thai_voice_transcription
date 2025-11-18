import nemo.collections.asr as nemo_asr
import torch

# Select processing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Typhoon ASR Real-Time model
print("Loading Typhoon ASR Real-Time...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="scb10x/typhoon-asr-realtime",
    map_location=device
)