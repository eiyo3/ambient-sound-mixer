# AmbientMaker Web - Phase 1
# Based on ambient_gui_v5.7.10 by 石川遼太郎
# Streamlit Web版

import streamlit as st
import numpy as np
import io
import wave
from scipy.signal import butter, lfilter, fftconvolve

st.set_page_config(page_title="AmbientMaker Web", page_icon="🎵", layout="wide")

# ========== Constants ==========
EXPORT_FS = 44100

# ========== DSP Utilities ==========

def norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-9)
    return (x / m).astype(np.float32)

def db2lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def fade_out_tail(x: np.ndarray, fs: int, fade_ms: float) -> np.ndarray:
    if x.size == 0:
        return x
    fadeN = int(round(fs * fade_ms / 1000.0))
    fadeN = max(1, min(fadeN, x.size))
    w = np.ones(x.size, dtype=np.float32)
    w[-fadeN:] = np.linspace(1.0, 0.0, fadeN, dtype=np.float32)
    return (x * w).astype(np.float32)

def trim_ir(ir: np.ndarray, fs: int, max_sec: float = 3.0, tail_fade_ms: float = 80.0) -> np.ndarray:
    if ir is None or ir.size <= 1:
        return ir
    maxN = int(round(fs * max_sec))
    if ir.size > maxN:
        ir = ir[:maxN].copy()
        ir = fade_out_tail(ir, fs, tail_fade_ms)
    return ir.astype(np.float32)

# ========== Sound Generators ==========

def make_wind(fs: int, dur: float) -> np.ndarray:
    n = int(fs * dur)
    white = np.random.randn(n).astype(np.float32)
    b, a = butter(2, 1000 / (fs / 2), 'low')
    wind = lfilter(b, a, white).astype(np.float32)
    t = np.arange(n) / fs
    lfo = (np.sin(2 * np.pi * 0.08 * t) * 0.5 + 0.5) * 0.3 + 0.7
    return norm(wind * lfo)

def make_air(fs: int, dur: float) -> np.ndarray:
    n = int(fs * dur)
    white = np.random.randn(n).astype(np.float32)
    b, a = butter(2, [2000 / (fs / 2), 12000 / (fs / 2)], btype='band')
    air = lfilter(b, a, white).astype(np.float32)
    return norm(air * 0.6)

def make_stream(fs: int, dur: float) -> np.ndarray:
    n = int(fs * dur)
    white = np.random.randn(n).astype(np.float32)
    b, a = butter(2, [300 / (fs / 2), 3000 / (fs / 2)], btype='band')
    s = lfilter(b, a, white).astype(np.float32)
    return norm(s * 0.8)

def make_ir_schroeder(fs: int, rt60: float = 5.0, predelay_ms: float = 30.0) -> np.ndarray:
    length = int(fs * rt60)
    ir = np.zeros(length, np.float32)
    for d, g in [(0.000, 1.00), (0.017, 0.55), (0.031, 0.45), (0.048, 0.35)]:
        idx = int(d * fs)
        if idx < len(ir):
            ir[idx] += g
    t = np.arange(length) / fs
    env = np.exp(-t * (6.0 / rt60)).astype(np.float32)
    b, a = butter(1, 6000 / (fs / 2), 'low')
    step = np.zeros(length, np.float32)
    step[0] = 1.0
    tail = lfilter(b, a, step).astype(np.float32)
    ir += tail * 0.9 * env
    pd = int(predelay_ms * fs / 1000)
    ir = np.concatenate([np.zeros(pd, np.float32), ir])
    return norm(ir)

# ========== RBJ Biquad ==========

def biquad_peaking(fc, Q, gain_db, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A; b1 = -2 * np.cos(w0); b2 = 1 - alpha * A
    a0 = 1 + alpha / A; a1 = -2 * np.cos(w0); a2 = 1 - alpha / A
    return (np.array([b0, b1, b2]) / a0).astype(np.float32), \
           np.array([1.0, a1 / a0, a2 / a0]).astype(np.float32)

def biquad_highshelf(fc, Q, gain_db, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cosw0 = np.cos(w0)
    b0 = A * ((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cosw0)
    b2 = A * ((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 = 2 * ((A-1) - (A+1)*cosw0)
    a2 = (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    return (np.array([b0, b1, b2]) / a0).astype(np.float32), \
           np.array([1.0, a1/a0, a2/a0]).astype(np.float32)

# ========== Motif Generator ==========

def generate_motif(fs, n_samples, bpm, density, scale, root, motif_vol):
    if motif_vol <= 0.0:
        return np.zeros(n_samples, np.float32)
    NOTE2SEMI = {"C":0,"C#":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,
                 "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
    root_semi = NOTE2SEMI.get(root, 9)
    triads_maj = [[0,4,7],[7,11,14],[9,12,16],[5,9,12]]
    triads_min = [[0,3,7],[7,10,14],[9,12,15],[5,8,12]]
    step_samples = int(round(fs * 60.0 / max(1e-6, bpm) / 2.0))
    out = np.zeros(n_samples, np.float32)
    pos, step = 0, 0
    while pos < n_samples:
        if np.random.rand() < density:
            bank = triads_maj if scale == "major" else triads_min
            degs = bank[(step // 8) % 4]
            add = np.random.choice([0,12,-12,24,-24], p=[0.6,0.15,0.15,0.05,0.05])
            semi = root_semi + int(np.random.choice(degs)) + add
            freq = 261.6256 * (2 ** ((semi - 12) / 12))
            decay = float(np.random.uniform(1.2, 3.0))
            ph = float(np.random.uniform(0, 2*np.pi))
            amp = float(np.random.uniform(0.7, 1.0))
            note_len = min(int(decay * 4 * fs), n_samples - pos)
            if note_len > 0:
                tt = np.arange(note_len, dtype=np.float32) / fs
                att_t = min(0.02, decay * 0.25)
                env_a = 0.5 - 0.5 * np.cos(np.pi * np.clip(tt / att_t, 0, 1))
                env_d = np.exp(-tt / decay)
                sig = (np.sin(2*np.pi*freq*tt + ph) +
                       0.30*np.sin(2*np.pi*2*freq*tt + ph)) * (env_a * env_d * amp)
                out[pos:pos+note_len] += sig.astype(np.float32)
        pos += step_samples
        step += 1
    b, a = butter(1, 120 / (fs/2), 'high')
    out = lfilter(b, a, out).astype(np.float32)
    return out * motif_vol

# ========== Main Synthesizer ==========

def synthesize_ambient(duration_sec, fs, drone_vol, drone_trim_db, noise_vol,
                       noise_type, reverb_wet, ir_type, cutoff_hz, lfo_rate,
                       lfo_to_reverb, lfo_to_cutoff, lfo_to_vol, hifi,
                       motif_on, motif_vol, motif_bpm, motif_density,
                       motif_scale, motif_root, master_db, progress_cb=None):
    total = int(fs * duration_sec)
    t = np.arange(total, dtype=np.float32) / fs

    # Drone
    drone = np.zeros(total, np.float32)
    for r in [1.0, 1.25, 1.5, 2.0]:
        drone += np.sin(2 * np.pi * 220.0 * r * t).astype(np.float32)
    drone *= (0.25 * drone_vol * db2lin(drone_trim_db))
    if progress_cb: progress_cb(0.15, "ドローン音を生成しました")

    # Noise
    noise_gen = {"風の音": make_wind, "空気の音": make_air, "水の音": make_stream,
                 "なし": lambda fs, d: np.zeros(int(fs*d), np.float32)}
    nraw = noise_gen.get(noise_type, noise_gen["なし"])(fs, duration_sec)
    if len(nraw) < total:
        nraw = np.tile(nraw, (total // len(nraw)) + 1)
    noise = nraw[:total] * noise_vol
    if progress_cb: progress_cb(0.30, "環境音を生成しました")

    # Motif
    motif = generate_motif(fs, total, motif_bpm, motif_density,
                           motif_scale, motif_root, motif_vol) if motif_on else np.zeros(total, np.float32)
    if progress_cb: progress_cb(0.40, "モチーフを生成しました")

    # LFO
    lfo_bi = np.sin(2 * np.pi * lfo_rate * t).astype(np.float32)
    lfo_uni = (lfo_bi * 0.5 + 0.5).astype(np.float32)
    drone *= (1.0 - lfo_uni * lfo_to_vol)

    # Mix
    sig = (drone + noise + motif).astype(np.float32)
    if progress_cb: progress_cb(0.50, "ミックスしました")

    # Cutoff LPF
    eff_cut = float(np.clip(cutoff_hz, 200, fs * 0.49))
    b, a = butter(2, eff_cut / (fs/2), 'low')
    sig = lfilter(b, a, sig).astype(np.float32)
    if progress_cb: progress_cb(0.60, "フィルタを適用しました")

    # Reverb
    if reverb_wet > 0:
        ir_gen = {"残響1": lambda: make_ir_schroeder(fs,6.0,40),
                  "残響2": lambda: make_ir_schroeder(fs,4.5,20),
                  "残響3": lambda: make_ir_schroeder(fs,2.5,0),
                  "なし": lambda: np.zeros(1, np.float32)}
        ir = trim_ir(ir_gen.get(ir_type, ir_gen["残響3"])(), fs)
        if ir.size > 1:
            wet = fftconvolve(sig, ir, mode='full')[:total].astype(np.float32)
            w = np.clip(reverb_wet + lfo_uni * lfo_to_reverb, 0, 1).astype(np.float32)
            sig = sig * (1.0 - w) + wet * w
    if progress_cb: progress_cb(0.75, "残響を適用しました")

    # Output HPF + Final LPF
    b, a = butter(2, 70 / (fs/2), 'high')
    sig = lfilter(b, a, sig).astype(np.float32)
    b, a = butter(1, 18000 / (fs/2), 'low')
    sig = lfilter(b, a, sig).astype(np.float32)

    # HiFi EQ
    if hifi:
        b, a = biquad_peaking(300.0, 0.8, -3.5, fs)
        sig = lfilter(b, a, sig).astype(np.float32)
        b, a = biquad_highshelf(8000.0, 0.7, 2.0, fs)
        sig = lfilter(b, a, sig).astype(np.float32)
    if progress_cb: progress_cb(0.85, "EQを適用しました")

    # Stereo (allpass decorrelation)
    g = 0.6
    R = lfilter(np.array([-g,1.0],np.float32), np.array([1.0,-g],np.float32), sig).astype(np.float32)
    L = sig.copy()

    # Master
    ml = db2lin(master_db)
    L *= ml; R *= ml

    # Fade in/out
    fi = int(fs * 0.02)
    if fi > 0 and fi < total:
        f = np.linspace(0,1,fi, dtype=np.float32)
        L[:fi] *= f; R[:fi] *= f
    fo = int(fs * 0.5)
    if fo > 0 and fo < total:
        f = np.linspace(1,0,fo, dtype=np.float32)
        L[-fo:] *= f; R[-fo:] *= f

    # Safety
    L = np.clip(np.nan_to_num(L), -1, 1).astype(np.float32)
    R = np.clip(np.nan_to_num(R), -1, 1).astype(np.float32)
    if progress_cb: progress_cb(0.95, "最終処理中...")

    return np.column_stack([L, R]).astype(np.float32)

def audio_to_wav(audio, fs):
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf

# ========== UI ==========

st.title("🎵 AmbientMaker Web")
st.caption("環境音生成ツール — ブラウザで自分だけのアンビエントサウンドを作ろう")
st.markdown("---")

# Macros
st.header("🎛️ 簡易設定")
mc1, mc2 = st.columns(2)
with mc1:
    m_tone = st.slider("☀️ 明るさ", 0.0, 1.0, 0.60, 0.01,
                        help="低い=暗く落ち着いた音、高い=明るく開けた音")
    m_tex = st.slider("🌿 音の質感", 0.0, 1.0, 0.00, 0.01,
                       help="低い=持続音中心、高い=環境音が増える")
with mc2:
    m_space = st.slider("🏛️ 残響の強さ", 0.0, 1.0, 0.60, 0.01,
                         help="低い=ドライ、高い=深い残響")
    m_motion = st.slider("🌊 変化（ゆらぎ）", 0.0, 1.0, 0.20, 0.01,
                          help="低い=静的、高い=ゆっくり変化")

d_cut = 2000 + m_tone * 10000
d_drone = 0.18 + 0.18 * (1.0 - m_tex)
d_noise = 0.25 * m_tex
d_rev = 0.08 + 0.82 * m_space
d_lfo = 0.05 + 0.25 * m_motion
d_lfo_rev = 0.10 * m_motion

st.markdown("---")

# Advanced
with st.expander("⚙️ 詳細設定"):
    ac1, ac2 = st.columns(2)
    with ac1:
        st.subheader("🔊 音量・音源")
        drone_vol = st.slider("持続音", 0.0, 1.0, round(d_drone, 2), 0.01, key="dv")
        drone_trim = st.slider("持続音調整(dB)", -24.0, 6.0, -9.0, 0.5, key="dt")
        noise_vol = st.slider("環境音", 0.0, 0.4, round(d_noise, 2), 0.01, key="nv")
        noise_type = st.selectbox("環境音の種類", ["風の音","空気の音","水の音","なし"], key="nt")
        st.subheader("🎹 モチーフ")
        motif_on = st.checkbox("モチーフON", True, key="mo")
        motif_vol = st.slider("モチーフ音量", 0.0, 0.5, 0.10, 0.01, key="mv")
        motif_bpm = st.slider("テンポ(BPM)", 40, 120, 70, key="mb")
        motif_density = st.slider("頻度", 0.0, 1.0, 0.25, 0.01, key="md")
        motif_scale = st.selectbox("音階", ["major","minor"], key="ms")
        motif_root = st.selectbox("基準音", ["A","Bb","B","C","C#","D","Eb","E","F","F#","G","G#"], key="mr")
    with ac2:
        st.subheader("🏛️ 残響・フィルタ")
        reverb_wet = st.slider("残響量", 0.0, 1.0, round(d_rev, 2), 0.01, key="rw")
        ir_type = st.selectbox("残響種類", ["残響1","残響2","残響3","なし"], key="it")
        cutoff_hz = st.slider("高音上限(Hz)", 200, 12000, int(d_cut), 10, key="ch")
        st.subheader("🌊 LFO")
        lfo_rate = st.slider("変化速さ(Hz)", 0.05, 2.0, round(d_lfo, 2), 0.01, key="lr")
        lfo_rev = st.slider("残響変化", 0.0, 1.0, round(d_lfo_rev, 2), 0.01, key="l2r")
        lfo_cut = st.slider("高音変化", 0.0, 1.0, 0.0, 0.01, key="l2c")
        lfo_vol = st.slider("音量変化", 0.0, 1.0, 0.0, 0.01, key="l2v")
        st.subheader("🎚️ マスター")
        hifi = st.checkbox("HiFi EQ", True, key="hf")
        master_db = st.slider("全体(dB)", -24.0, 6.0, -6.0, 0.5, key="mdb")

# Defaults if advanced not touched
if "dv" not in st.session_state:
    drone_vol = d_drone; drone_trim = -9.0; noise_vol = d_noise
    noise_type = "風の音"; reverb_wet = d_rev; ir_type = "残響3"
    cutoff_hz = int(d_cut); lfo_rate = d_lfo; lfo_rev = d_lfo_rev
    lfo_cut = 0.0; lfo_vol = 0.0; hifi = True; master_db = -6.0
    motif_on = True; motif_vol = 0.10; motif_bpm = 70
    motif_density = 0.25; motif_scale = "major"; motif_root = "A"

st.markdown("---")

# Generate
st.header("🎵 サウンド生成")
gc1, gc2 = st.columns(2)
with gc1:
    dur = st.selectbox("生成する長さ", [10,30,60,120,180,240,300], index=2,
                       format_func=lambda x: f"{x}秒 ({x//60}分{x%60}秒)" if x>=60 else f"{x}秒")
with gc2:
    st.write(""); st.write("")
    go = st.button("🎵 サウンドを生成する", type="primary", use_container_width=True)

if go:
    prog = st.progress(0)
    stat = st.empty()
    def cb(p, m):
        prog.progress(min(p, 1.0)); stat.text(f"⏳ {m}")
    cb(0.05, "生成を開始します...")

    audio = synthesize_ambient(
        dur, EXPORT_FS, drone_vol, drone_trim, noise_vol, noise_type,
        reverb_wet, ir_type, cutoff_hz, lfo_rate, lfo_rev, lfo_cut, lfo_vol,
        hifi, motif_on, motif_vol, motif_bpm, motif_density, motif_scale,
        motif_root, master_db, cb)

    prog.progress(1.0); stat.text("✅ 生成完了！")
    wav = audio_to_wav(audio, EXPORT_FS)
    st.success("✅ サウンドが生成されました！")
    wav.seek(0); st.audio(wav, format="audio/wav")
    wav.seek(0)
    st.download_button("💾 WAVファイルをダウンロード", wav,
                       f"ambient_{dur}s.wav", "audio/wav", use_container_width=True)

st.markdown("---")
st.caption("AmbientMaker Web — Phase 1 | Created by 石川遼太郎")
