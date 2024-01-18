# from talklip project: https://github.com/Sxjdwang/TalkLip/blob/main/train.py#L213
from python_speech_features import logfbank
from scipy.io import wavfile
import numpy as np
def calculate_mel(wav_data, sample_rate):
    # calculate mel: 3.45s -> (345 x 26)
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F], (345, 26) -> 3.45s
    return audio_feats

wav_path = "/ws-laion/CMLR_quick_view/s2/20180302/section_5_012.81_016.27/audio.wav" #  3.45s
sampRate, wav = wavfile.read(wav_path)
audio_feat = calculate_mel(wav, sampRate) # (345, 26)
print(audio_feat.shape)
