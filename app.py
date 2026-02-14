import streamlit as st
import numpy as np
import io
import wave
import struct

st.set_page_config(page_title="Ambient Sound Mixer", page_icon="🎵")

st.title("🎵 Ambient Sound Mixer")
st.write("スライダーを調整して、自分だけのアンビエントサウンドを作ろう！")

# --- パラメータ設定 ---
st.sidebar.header("⚙️ 設定")
duration = st.sidebar.slider("再生時間（秒）", 1, 10, 3)
sample_rate = 22050

st.header("🎛️ サウンドミキサー")

col1, col2 = st.columns(2)

with col1:
    rain_vol = st.slider("🌧️ 雨の音", 0.0, 1.0, 0.3)
    wind_vol = st.slider("💨 風の音", 0.0, 1.0, 0.2)

with col2:
    wave_vol = st.slider("🌊 波の音", 0.0, 1.0, 0.2)
    bird_vol = st.slider("🐦 鳥の声", 0.0, 1.0, 0.1)


# --- サウンド生成関数 ---
def generate_noise(n_samples):
    """ホワイトノイズ（雨の音のベース）"""
    return np.random.randn(n_samples) * 0.3


def generate_wind(n_samples, sr):
    """風の音（低周波ノイズ）"""
    noise = np.random.randn(n_samples)
    # 簡易ローパスフィルタ
    filtered = np.zeros(n_samples)
    alpha = 0.02
    filtered[0] = noise[0]
    for i in range(1, n_samples):
        filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i - 1]
    # 音量の揺らぎを追加
    t = np.linspace(0, duration, n_samples)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    return filtered * envelope * 5


def generate_waves(n_samples, sr):
    """波の音（周期的なノイズ）"""
    t = np.linspace(0, duration, n_samples)
    noise = np.random.randn(n_samples) * 0.3
    # 波の周期的な音量変化
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    return noise * envelope


def generate_birds(n_samples, sr):
    """鳥の声（高周波のチャープ音）"""
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    # ランダムなタイミングで鳥の声を配置
    np.random.seed(42)
    n_chirps = int(duration * 2)
    for _ in range(n_chirps):
        start = np.random.randint(0, max(1, n_samples - sr // 4))
        chirp_len = np.random.randint(sr // 20, sr // 8)
        end = min(start + chirp_len, n_samples)
        chirp_t = np.linspace(0, 1, end - start)
        freq = np.random.uniform(2000, 4000)
        chirp = np.sin(2 * np.pi * freq * chirp_t) * np.exp(-3 * chirp_t)
        signal[start:end] += chirp * 0.5
    return signal


def mix_to_wav(audio, sr):
    """NumPy配列をWAVバイトに変換"""
    # 正規化
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    # 16bit整数に変換
    audio_int16 = (audio * 32767).astype(np.int16)
    # WAVファイルをメモリに書き出し
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer


# --- 生成ボタン ---
if st.button("🎵 サウンドを生成する", type="primary"):
    with st.spinner("生成中..."):
        n_samples = sample_rate * duration

        # 各サウンドを生成してミックス
        mixed = np.zeros(n_samples)
        if rain_vol > 0:
            mixed += generate_noise(n_samples) * rain_vol
        if wind_vol > 0:
            mixed += generate_wind(n_samples, sample_rate) * wind_vol
        if wave_vol > 0:
            mixed += generate_waves(n_samples, sample_rate) * wave_vol
        if bird_vol > 0:
            mixed += generate_birds(n_samples, sample_rate) * bird_vol

        # WAVに変換
        wav_buffer = mix_to_wav(mixed, sample_rate)

        st.success("✅ 生成完了！")
        st.audio(wav_buffer, format="audio/wav")

        # ダウンロードボタン
        wav_buffer.seek(0)
        st.download_button(
            label="💾 WAVファイルをダウンロード",
            data=wav_buffer,
            file_name="ambient_mix.wav",
            mime="audio/wav",
        )

# --- フッター ---
st.markdown("---")
st.caption("Ambient Sound Mixer - Streamlit Demo App")
