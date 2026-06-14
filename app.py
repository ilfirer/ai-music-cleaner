import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from pydub import AudioSegment
import matplotlib.pyplot as plt
import io
import os
import tempfile

st.set_page_config(page_title="AI Music Cleaner", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> AI Music Cleaner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Очистка музыки от артефактов нейросетей (Suno, Udio, Riffusion)</p>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Настройки обработки")

humanize_enabled = st.sidebar.checkbox("🎭 Humanization", value=True)
timing_strength = st.sidebar.slider("Сила сдвига тайминга", 0.0, 1.0, 0.5, 0.1) if humanize_enabled else 0.0
pitch_strength = st.sidebar.slider("Сила модуляции питча", 0.0, 1.0, 0.3, 0.1) if humanize_enabled else 0.0
warmth_drive = st.sidebar.slider("Аналоговая сатурация", 1.0, 2.0, 1.2, 0.1) if humanize_enabled else 1.0
swing_percent = st.sidebar.slider("Swing %", 0, 20, 10, 1) if humanize_enabled else 0

anti_forensics_enabled = st.sidebar.checkbox("🛡️ Anti-Forensics", value=True)
anti_intensity = st.sidebar.slider("Интенсивность anti-forensics", 0.0, 1.0, 0.5, 0.1) if anti_forensics_enabled else 0.0
pitch_shift_percent = st.sidebar.slider("Pitch shift %", 0.0, 1.0, 0.3, 0.1) if anti_forensics_enabled else 0.0

mastering_enabled = st.sidebar.checkbox("🎚️ Мастеринг", value=True)
compression_threshold = st.sidebar.slider("Порог компрессии (dB)", -24, 0, -12, 1) if mastering_enabled else -12
compression_ratio = st.sidebar.slider("Ratio компрессии", 1.0, 4.0, 2.0, 0.5) if mastering_enabled else 1.0

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

def humanize_timing(audio, sr, strength=0.5):
    if strength == 0:
        return audio
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, backtrack=True)
    onset_samples = librosa.frames_to_samples(onset_frames)
    audio_out = audio.copy()
    for i in range(len(onset_samples) - 1):
        start = onset_samples[i]
        end = onset_samples[i + 1] if i + 1 < len(onset_samples) else len(audio)
        shift_samples = int(np.random.normal(0, 0.015 * strength * sr))
        shift_samples = np.clip(shift_samples, -int(0.015 * sr), int(0.015 * sr))
        if shift_samples != 0:
            segment = audio[start:end]
            if shift_samples > 0:
                audio_out[start + shift_samples:end + shift_samples] = segment[:len(segment) - shift_samples]
            else:
                audio_out[start:end + shift_samples] = segment[-shift_samples:]
    return audio_out

def humanize_pitch(audio, sr, strength=0.3):
    if strength == 0:
        return audio
    n_steps = np.random.uniform(-0.1, 0.1) * strength
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def add_analog_warmth(audio, drive=1.2):
    return np.tanh(audio * drive) / np.tanh(drive)

def add_swing(audio, sr, swing_percent=10):
    if swing_percent == 0:
        return audio
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_samples = librosa.frames_to_samples(onset_frames)
    audio_out = audio.copy()
    for i in range(1, len(onset_samples) - 1, 2):
        start = onset_samples[i]
        next_onset = onset_samples[i + 1] if i + 1 < len(onset_samples) else len(audio)
        duration = next_onset - start
        shift = int(duration * swing_percent / 100)
        if shift > 0 and start + shift < len(audio):
            segment = audio[start:next_onset]
            audio_out[start + shift:start + shift + len(segment)] = segment
    return audio_out

def break_spectral_watermarks(audio, sr, pitch_shift_percent=0.3):
    if pitch_shift_percent == 0:
        return audio
    n_steps = pitch_shift_percent * 0.1
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def add_ultrasonic_noise(audio, sr, level_db=-60):
    noise = np.random.normal(0, 1, len(audio))
    nyq = 0.5 * sr
    cutoff = 18000 / nyq
    b, a = signal.butter(4, cutoff, btype='high')
    noise_filtered = signal.lfilter(b, a, noise)
    noise_rms = np.sqrt(np.mean(noise_filtered ** 2))
    target_rms = 10 ** (level_db / 20)
    noise_normalized = noise_filtered * (target_rms / noise_rms)
    return audio + noise_normalized

def analog_degradation(audio, sr, intensity=0.3):
    if intensity == 0:
        return audio
    noise = np.random.normal(0, 1, len(audio))
    b, a = signal.butter(2, 0.01, btype='low')
    pink_noise = signal.lfilter(b, a, noise)
    noise_rms = np.sqrt(np.mean(pink_noise ** 2))
    target_rms = 10 ** (-40 / 20) * intensity
    pink_noise_normalized = pink_noise * (target_rms / noise_rms)
    return audio + pink_noise_normalized

def apply_compression(audio, threshold_db=-12, ratio=2.0):
    threshold_linear = 10 ** (threshold_db / 20)
    audio_abs = np.abs(audio)
    gain_reduction = np.where(audio_abs > threshold_linear, threshold_linear + (audio_abs - threshold_linear) / ratio, audio_abs)
    audio_compressed = np.sign(audio) * gain_reduction
    audio_compressed *= ratio ** 0.5
    return audio_compressed

def apply_eq(audio, sr):
    nyq = 0.5 * sr
    b_hp, a_hp = signal.butter(2, 30 / nyq, btype='high')
    audio_eq = signal.lfilter(b_hp, a_hp, audio)
    b_ls, a_ls = signal.butter(2, 200 / nyq, btype='low')
    low_freq = signal.lfilter(b_ls, a_ls, audio_eq)
    audio_eq = audio_eq + low_freq * 0.26
    b_hs, a_hs = signal.butter(2, 10000 / nyq, btype='high')
    high_freq = signal.lfilter(b_hs, a_hs, audio_eq)
    audio_eq = audio_eq + high_freq * 0.19
    return audio_eq

def apply_limiting(audio, ceiling_db=-1.0):
    ceiling_linear = 10 ** (ceiling_db / 20)
    return np.clip(audio, -ceiling_linear, ceiling_linear)

def plot_spectrogram(audio, sr, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    return fig

uploaded_file = st.file_uploader("📁 Загрузите аудиофайл (WAV, MP3, FLAC)", type=['wav', 'mp3', 'flac'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        input_path = tmp_file.name
    
    st.success(f"✅ Файл загружен: {uploaded_file.name}")
    
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=0)
    else:
        audio_mono = audio
    
    st.info(f"📊 Информация: {sr} Hz, {len(audio_mono)/sr:.2f} сек")
    
    st.subheader(" Оригинальный спектр")
    st.pyplot(plot_spectrogram(audio_mono, sr, "Оригинал"))
    
    if st.button("🚀 Обработать трек", type="primary"):
        with st.spinner("⏳ Обработка..."):
            audio_processed = audio_mono.copy()
            
            if humanize_enabled:
                audio_processed = humanize_timing(audio_processed, sr, timing_strength)
                audio_processed = humanize_pitch(audio_processed, sr, pitch_strength)
                audio_processed = add_analog_warmth(audio_processed, warmth_drive)
                audio_processed = add_swing(audio_processed, sr, swing_percent)
            
            if anti_forensics_enabled:
                audio_processed = break_spectral_watermarks(audio_processed, sr, pitch_shift_percent)
                audio_processed = add_ultrasonic_noise(audio_processed, sr, -60)
                audio_processed = analog_degradation(audio_processed, sr, anti_intensity)
            
            if mastering_enabled:
                audio_processed = apply_eq(audio_processed, sr)
                audio_processed = apply_compression(audio_processed, compression_threshold, compression_ratio)
                audio_processed = apply_limiting(audio_processed, -1.0)
            
            peak = np.max(np.abs(audio_processed))
            if peak > 0:
                audio_processed = audio_processed * (0.99 / peak)
            
            st.subheader("📈 Обработанный спектр")
            st.pyplot(plot_spectrogram(audio_processed, sr, "После обработки"))
            
            output_path = input_path.replace(os.path.splitext(input_path)[1], '_cleaned.wav')
            sf.write(output_path, audio_processed, sr, subtype='PCM_16')
            
            with open(output_path, 'rb') as f:
                audio_bytes = f.read()
            
            st.success("✅ Обработка завершена!")
            
            st.download_button(
                label="💾 Скачать очищенный файл",
                data=audio_bytes,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_cleaned.wav",
                mime="audio/wav",
                type="primary"
            )
            
            os.unlink(input_path)
            os.unlink(output_path)

st.markdown("---")
st.markdown("""
### ℹ️ О приложении
**AI Music Cleaner** — инструмент для очистки музыки от артефактов нейросетей (Suno, Udio, Riffusion).

- 🎭 **Humanization**: человеческий грув
- 🛡️ **Anti-Forensics**: удаление водяных знаков
- 🎚️ **Мастеринг**: эквализация и компрессия
""")
