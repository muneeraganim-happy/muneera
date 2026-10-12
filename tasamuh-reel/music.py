"""Synthesise a calm piano + strings background track (45 s, D major, 72 BPM)."""
import numpy as np, wave, sys

SR, DUR = 44100, 45.0
N = int(SR * DUR)
out = np.zeros((N, 2))
beat = 60 / 72
bar = beat * 4

def hz(m): return 440 * 2 ** ((m - 69) / 12)

def piano(m, t0, vel, length=3.0, pan=0.0):
    i0 = int(t0 * SR); n = min(int(length * SR), N - i0)
    if n <= 0: return
    t = np.arange(n) / SR; f = hz(m)
    s = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t * (1.2 + k * 0.9)) for k, a in [(1, 1), (2, .45), (3, .18), (4, .08)])
    s *= np.minimum(1, t / 0.006) * vel
    out[i0:i0 + n, 0] += s * (1 - pan) * 0.5
    out[i0:i0 + n, 1] += s * (1 + pan) * 0.5

def pad(ms, t0, length, vel):
    i0 = int(t0 * SR); n = min(int(length * SR), N - i0)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 1.2) * np.minimum(1, (length - t) / 1.2)
    for m in ms:
        for det, ch in [(-0.08, 0), (0.08, 1)]:
            f = hz(m + det)
            s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
            out[i0:i0 + n, ch] += s * env * vel * (1 + 0.15 * np.sin(2 * np.pi * 0.25 * t))

# D, Bm, G, A, D/F#, G, Em7, A, D, Bm, G, A, D
prog = [[50, 57, 62, 66], [47, 54, 59, 62], [43, 50, 55, 59], [45, 52, 57, 61],
        [42, 50, 57, 62], [43, 50, 55, 59], [40, 47, 55, 62], [45, 52, 57, 61],
        [50, 57, 62, 66], [47, 54, 59, 62], [43, 50, 55, 59], [45, 52, 57, 61], [50, 57, 62, 66]]
melody = {0: [74, None, 73, 71], 1: [71, None, 69, None], 2: [67, 69, 71, None], 3: [69, None, None, None],
          4: [74, None, 76, 78], 5: [79, None, 78, 76], 6: [74, None, 76, None], 7: [73, None, None, None],
          8: [78, None, 76, 74], 9: [74, None, 73, 71], 10: [71, 73, 74, None], 11: [76, None, 73, None], 12: [74, None, None, None]}

for b, ch in enumerate(prog):
    t0 = b * bar
    swell = 1.0 + 0.5 * np.exp(-((b - 7.5) / 2.2) ** 2)          # lift around the hug (~23-29 s)
    last = b == len(prog) - 1
    piano(ch[0] - 12, t0, 0.28, 6 if last else 4, -0.2)
    if not last:
        arp = [ch[1], ch[2], ch[3], ch[2] + 12, ch[3], ch[2], ch[1], ch[2]]
        for k, m in enumerate(arp):
            piano(m, t0 + k * beat / 2, 0.11 * swell * (1.1 if k == 0 else 1), 2.5, 0.3 * np.sin(k))
    else:
        for k, m in enumerate(ch[1:] + [ch[3] + 12]):
            piano(m, t0 + k * 0.12, 0.13, 6, 0.2 * k - 0.3)
    if b >= 1:
        pad([m + 12 for m in ch[1:]], t0, bar + 1.2, 0.012 * swell)
    for k, m in enumerate(melody.get(b, [])):
        if m and b >= 2:
            piano(m, t0 + k * beat, 0.16 * swell, 3.5, 0.1)

# simple reverb: convolve with a decaying noise impulse response
rng = np.random.default_rng(7)
L = int(SR * 2.8)
ir = rng.standard_normal((L, 2)) * np.exp(-np.arange(L) / SR * 2.4)[:, None]
ir[:int(SR * 0.02)] = 0
ir /= np.sqrt((ir ** 2).sum(0))
M = 1 << int(np.ceil(np.log2(N + L)))
wet = np.stack([np.fft.irfft(np.fft.rfft(out[:, c], M) * np.fft.rfft(ir[:, c], M), M)[:N] for c in range(2)], 1)
mix = out * 0.75 + wet * 0.55

t = np.arange(N) / SR
mix *= np.minimum(1, t / 1.5)[:, None] * np.clip((DUR - t) / 3.0, 0, 1)[:, None]
mix /= np.abs(mix).max() / 0.8
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
