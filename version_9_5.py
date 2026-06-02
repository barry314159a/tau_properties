# SEGMENTED FOURIER ANALYSIS OF MINIMUM MODULI  (v9.5)
#
# KEY CHANGE: Instead of fitting one curve to the entire dataset, the code
# first detects breakpoints where the data's behavior changes (e.g., from
# a linear ramp to a flat plateau), splits the data into segments at those
# breakpoints, and detrends each segment independently with a simple
# linear fit.  The periodicity analysis then runs on each segment separately.
#
# This avoids the fundamental problem of all previous versions: no single
# polynomial or smoother can cleanly follow a piecewise-linear shape, and
# the residual artifacts from trying corrupt every downstream step.
#
# It expects mins_list to already be defined.

import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PREPARE THE DATA
# ==============================================================================
indices   = np.array([float(pair[0]) for pair in mins_list], dtype=np.float64)
minmoduli = np.array([float(pair[1]) for pair in mins_list], dtype=np.float64)
N = len(indices)

print(f"Data range: n = {int(indices[0])} to {int(indices[-1])}")
print(f"Number of data points: {N}")
print(f"Min modulus range: {minmoduli.min():.6f} to {minmoduli.max():.6f}")

# ==============================================================================
# 2. BREAKPOINT DETECTION
# ==============================================================================
# Detect points where the data's local behavior changes qualitatively.
# We use a piecewise-linear model: the data is approximated by straight
# line segments joined at breakpoints.  The algorithm finds the breakpoint
# positions that minimize the total squared error.
#
# We use the 'ruptures' library (Truong, Oudre, Vayatis 2020) with the
# Pelt algorithm, which automatically determines the number of breakpoints.

import ruptures as rpt

# Penalty controls sensitivity: higher = fewer breakpoints, lower = more.
# We calibrate it relative to the data's variance.
penalty_value = np.log(N) * np.var(minmoduli)

algo = rpt.Pelt(model="l2", min_size=max(20, N // 10)).fit(minmoduli)
breakpoints = algo.predict(pen=penalty_value)

# ruptures returns breakpoint indices where each segment ENDS.
# The last entry is always N (end of data).  Convert to segment boundaries.
seg_boundaries = [0] + breakpoints  # e.g. [0, 120, 350]

n_segments = len(seg_boundaries) - 1

print(f"\n{'='*75}")
print(f"SEGMENTATION")
print(f"{'='*75}")
print(f"  Breakpoints detected: {n_segments - 1}")
for i in range(n_segments):
    start_idx = seg_boundaries[i]
    end_idx = seg_boundaries[i+1]
    seg_indices = indices[start_idx:end_idx]
    seg_values = minmoduli[start_idx:end_idx]
    print(f"  Segment {i+1}: n = {int(seg_indices[0])} to {int(seg_indices[-1])}  "
          f"({end_idx - start_idx} points)  "
          f"range [{seg_values.min():.2f}, {seg_values.max():.2f}]")

# ==============================================================================
# 3. ANALYZE EACH SEGMENT
# ==============================================================================

def analyze_segment(seg_indices, seg_values, seg_label):
    """Run the full periodicity analysis on one segment."""
    n = len(seg_indices)

    if n < 10:
        print(f"\n  Segment too short ({n} points) for analysis.")
        return None

    max_credible_period = n / 3.0

    # --- Linear detrend (appropriate for each piecewise-linear segment) ---
    coeffs = np.polyfit(seg_indices, seg_values, deg=1)
    trend = np.polyval(coeffs, seg_indices)
    detrended = seg_values - trend
    slope = coeffs[0]

    print(f"\n  Detrending: linear fit (slope = {slope:.4f})")
    print(f"    Detrended std dev: {np.std(detrended):.4f}")

    # --- Envelope normalization (if needed) ---
    abs_det = np.abs(detrended)
    env_win = max(5, min(n // 10, n // 3))
    if env_win % 2 == 0:
        env_win += 1
    env_win = min(env_win, n - 1)
    if env_win % 2 == 0:
        env_win -= 1
    if env_win < 5:
        env_win = 5
    if env_win >= n:
        envelope_applied = False
        analysis_signal = detrended
    else:
        envelope = signal.savgol_filter(abs_det, window_length=env_win,
                                        polyorder=min(2, env_win - 1))
        env_floor = np.percentile(abs_det[abs_det > 0], 5) if np.any(abs_det > 0) else 1.0
        envelope = np.maximum(envelope, env_floor)

        env_p10 = np.percentile(envelope, 10)
        env_p90 = np.percentile(envelope, 90)
        envelope_ratio = env_p90 / env_p10 if env_p10 > 0 else 1.0

        if envelope_ratio > 3.0:
            analysis_signal = detrended / envelope
            envelope_applied = True
            print(f"    Envelope normalization: APPLIED (dynamic range {envelope_ratio:.1f}x)")
        else:
            analysis_signal = detrended
            envelope_applied = False
            print(f"    Envelope normalization: not needed (dynamic range {envelope_ratio:.1f}x)")

    # --- Windowed FFT ---
    window = np.hanning(n)
    windowed = analysis_signal * window
    window_power_correction = n / np.sum(window**2)

    fft_result     = np.fft.fft(windowed)
    fft_freqs      = np.fft.fftfreq(n, d=1.0)
    power_spectrum = np.abs(fft_result)**2 * window_power_correction

    pos = fft_freqs > 0
    frequencies = fft_freqs[pos]
    power       = power_spectrum[pos]
    periods     = 1.0 / frequencies

    # --- AR(1) significance with Bonferroni correction ---
    rho = np.corrcoef(analysis_signal[:-1], analysis_signal[1:])[0, 1]
    variance = np.var(analysis_signal)

    ar1_power = (variance * (1 - rho**2)) / (
        1 - 2 * rho * np.cos(2 * np.pi * frequencies) + rho**2
    )
    ar1_power *= np.mean(power) / np.mean(ar1_power)

    credible_mask = periods <= max_credible_period
    n_tests = int(np.sum(credible_mask))
    if n_tests < 1:
        n_tests = 1

    alpha_95_corrected = 1.0 - (0.05 / n_tests)
    alpha_99_corrected = 1.0 - (0.01 / n_tests)

    chi2_95 = stats.chi2.ppf(alpha_95_corrected, df=2) / 2.0
    chi2_99 = stats.chi2.ppf(alpha_99_corrected, df=2) / 2.0
    conf_95 = ar1_power * chi2_95
    conf_99 = ar1_power * chi2_99

    # --- Significant FFT peaks ---
    peak_idx = signal.find_peaks(power, height=0)[0]
    significant_peaks = []
    for pi in peak_idx:
        if not credible_mask[pi]:
            continue
        if power[pi] > conf_99[pi]:
            significant_peaks.append((pi, '99%'))
        elif power[pi] > conf_95[pi]:
            significant_peaks.append((pi, '95%'))

    significant_peaks.sort(key=lambda x: power[x[0]], reverse=True)

    # --- Multi-scale ACF fundamental detection ---
    detected_periods = []
    smooth_widths = [1]
    w = 3
    while w < max_credible_period / 2:
        smooth_widths.append(w)
        w = max(w + 2, int(w * 1.5))
        if w % 2 == 0:
            w += 1

    for sw in smooth_widths:
        if sw == 1:
            smoothed = analysis_signal.copy()
            scale_label = "raw"
        else:
            kernel = np.ones(sw) / sw
            smoothed = np.convolve(analysis_signal, kernel, mode='same')
            scale_label = f"MA({sw})"

        sm_centered = smoothed - smoothed.mean()
        acf_sm = np.correlate(sm_centered, sm_centered, 'full')
        acf_sm = acf_sm[n-1:]
        if acf_sm[0] > 0:
            acf_sm = acf_sm / acf_sm[0]
        else:
            continue

        search_start = max(2, sw + 1) if sw > 1 else 2
        search_end = min(int(max_credible_period), len(acf_sm) - 1)
        if search_start >= search_end:
            continue

        slice_begin = max(0, search_start - 1)
        acf_slice = acf_sm[slice_begin:search_end+1]
        peaks_in_slice, props = signal.find_peaks(acf_slice, height=0.1)

        boundary_offset = search_start - slice_begin
        if (boundary_offset not in peaks_in_slice and
                search_start < len(acf_sm) - 1 and search_start > 0):
            val = acf_sm[search_start]
            left = acf_sm[search_start - 1]
            right = acf_sm[search_start + 1]
            if val > left and val > right and val > 0.1:
                peaks_in_slice = np.concatenate(([boundary_offset], peaks_in_slice))

        if len(peaks_in_slice) > 0:
            true_lags = peaks_in_slice + slice_begin
            true_lags = true_lags[true_lags >= search_start]
            if len(true_lags) > 0:
                lag = int(true_lags[0])
                strength = acf_sm[lag]
                is_new = True
                for (prev_p, _, _) in detected_periods:
                    if abs(lag - prev_p) / max(lag, prev_p) < 0.15:
                        is_new = False
                        break
                if is_new:
                    detected_periods.append((float(lag), scale_label, float(strength)))

    detected_periods.sort(key=lambda x: x[0])

    fund_period = None
    fund_method = None
    fund_strength = None
    sub_periods = []
    best_fund_idx = 0

    if detected_periods:
        best_explained = 0
        for i, (cand_p, cand_label, cand_str) in enumerate(detected_periods):
            if cand_p < 2:
                continue
            n_explained = 0
            for j, (other_p, _, _) in enumerate(detected_periods):
                if j == i:
                    n_explained += 1
                    continue
                ratio = other_p / cand_p
                nearest = round(ratio)
                if nearest >= 1 and abs(ratio - nearest) / max(nearest, 1) < 0.15:
                    n_explained += 1
            if n_explained > best_explained or (
                    n_explained == best_explained and
                    cand_str > detected_periods[best_fund_idx][2]):
                best_explained = n_explained
                best_fund_idx = i

        fund_period = detected_periods[best_fund_idx][0]
        fund_method = f"multi-scale ACF ({detected_periods[best_fund_idx][1]})"
        fund_strength = detected_periods[best_fund_idx][2]
        sub_periods = [dp for k, dp in enumerate(detected_periods) if k != best_fund_idx]
        sub_periods.sort(key=lambda x: x[0], reverse=True)

    # --- Report ---
    has_fft_evidence = len(significant_peaks) > 0
    has_acf_evidence = fund_period is not None and fund_strength > 0.5

    print(f"\n    FFT significant peaks: {len(significant_peaks)}")
    if significant_peaks:
        print(f"    {'Rank':<6} {'Period':<10} {'Power':<15} {'Signif.':<10}")
        for rank, (pi, level) in enumerate(significant_peaks[:10], 1):
            print(f"    {rank:<6} {periods[pi]:>8.2f} {power[pi]:>12.2e}   {level}")

    if detected_periods:
        print(f"\n    Multi-scale periods detected:")
        for i, (p, label, strength) in enumerate(detected_periods):
            role = "FUNDAMENTAL" if i == best_fund_idx else "integer multiple"
            print(f"      {p:>6.1f}  {label:<12} ACF={strength:.4f}  {role}")

    # --- Verdict ---
    # Binary: PERIODIC or NOT PERIODIC.
    # PERIODIC requires ACF strength > 0.5.  The FFT result is noted when
    # it agrees but is never sufficient on its own — a single spectral peak
    # can clear the Bonferroni threshold in a noisy segment by chance, and
    # moderate ACF values (0.2–0.5) can arise from short-range noise
    # correlation rather than genuine periodicity.
    if has_acf_evidence:
        verdict = "PERIODIC"
        verdict_detail = f"period {fund_period:.1f}, ACF strength {fund_strength:.4f}"
        if has_fft_evidence:
            verdict_detail += "  (confirmed by FFT)"
    else:
        verdict = "NOT PERIODIC"
        verdict_detail = ""

    print(f"\n    Verdict: {verdict}")
    if verdict_detail:
        print(f"    {verdict_detail}")

    return {
        'verdict': verdict,
        'fund_period': fund_period,
        'fund_strength': fund_strength,
        'significant_peaks': significant_peaks,
        'seg_indices': seg_indices,
        'seg_values': seg_values,
        'trend': trend,
        'detrended': detrended,
        'analysis_signal': analysis_signal,
        'frequencies': frequencies,
        'power': power,
        'periods': periods,
        'ar1_power': ar1_power,
        'conf_95': conf_95,
        'conf_99': conf_99,
    }

# ==============================================================================
# RUN ANALYSIS ON EACH SEGMENT
# ==============================================================================

results = []
for i in range(n_segments):
    start_idx = seg_boundaries[i]
    end_idx = seg_boundaries[i+1]
    seg_idx = indices[start_idx:end_idx]
    seg_val = minmoduli[start_idx:end_idx]
    label = f"Segment {i+1}"

    print(f"\n{'='*75}")
    print(f"ANALYSIS: {label}  (n = {int(seg_idx[0])} to {int(seg_idx[-1])}, "
          f"{len(seg_idx)} points)")
    print(f"{'='*75}")

    result = analyze_segment(seg_idx, seg_val, label)
    results.append(result)

# ==============================================================================
# OVERALL SUMMARY
# ==============================================================================
print(f"\n{'='*75}")
print(f"OVERALL SUMMARY")
print(f"{'='*75}")
for i in range(n_segments):
    start_idx = seg_boundaries[i]
    end_idx = seg_boundaries[i+1]
    seg_idx = indices[start_idx:end_idx]
    r = results[i]
    if r is not None:
        print(f"  Segment {i+1} (n={int(seg_idx[0])}..{int(seg_idx[-1])}): {r['verdict']}", end="")
        if r['fund_period'] is not None:
            print(f"  period={r['fund_period']:.1f}  ACF={r['fund_strength']:.4f}", end="")
        print()
    else:
        print(f"  Segment {i+1}: too short")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# --- Overview plot: original data with breakpoints ---
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

axes[0].plot(indices, minmoduli, 'b-', linewidth=0.8)
for bp in seg_boundaries[1:-1]:
    axes[0].axvline(x=indices[bp] if bp < N else indices[-1],
                    color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Index n')
axes[0].set_ylabel('Minimum Modulus')
axes[0].set_title('Original Data with Detected Breakpoints')
axes[0].grid(True, alpha=0.3)

# Detrended segments stitched together for display
colors = ['blue', 'green', 'purple', 'orange', 'brown']
for i, r in enumerate(results):
    if r is not None:
        c = colors[i % len(colors)]
        axes[1].plot(r['seg_indices'], r['detrended'], '-', color=c,
                     linewidth=0.8, label=f'Segment {i+1}')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Index n')
axes[1].set_ylabel('Detrended')
axes[1].set_title('Segment-wise Detrended Residuals')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Per-segment detail plots ---
for i, r in enumerate(results):
    if r is None:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    seg_idx = r['seg_indices']

    # Original segment with linear trend
    axes[0, 0].plot(seg_idx, r['seg_values'], 'b-', linewidth=0.8, label='Data')
    axes[0, 0].plot(seg_idx, r['trend'], 'r-', linewidth=1.5, label='Linear trend')
    axes[0, 0].set_title(f'Segment {i+1}: Data with Linear Trend')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Analysis signal
    axes[0, 1].plot(seg_idx, r['analysis_signal'], 'b-', linewidth=0.8)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Segment {i+1}: Analysis Signal')
    axes[0, 1].grid(True, alpha=0.3)

    # Power spectrum
    freq = r['frequencies']
    pwr = r['power']
    axes[1, 0].semilogy(freq, pwr, 'b-', linewidth=0.8, label='Power')
    axes[1, 0].semilogy(freq, r['ar1_power'], 'k--', linewidth=1,
                        alpha=0.5, label='AR(1)')
    axes[1, 0].semilogy(freq, r['conf_95'], 'g--', linewidth=1,
                        alpha=0.6, label='95%')
    axes[1, 0].semilogy(freq, r['conf_99'], 'r--', linewidth=1,
                        alpha=0.6, label='99%')
    for pi, level in r['significant_peaks']:
        color = 'red' if level == '99%' else 'orange'
        axes[1, 0].semilogy(freq[pi], pwr[pi], 'o', color=color, markersize=8)
    axes[1, 0].set_title(f'Segment {i+1}: Power Spectrum')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 0.5)

    # Power spectrum vs period
    periods = r['periods']
    n_seg = len(seg_idx)
    max_p = n_seg / 3.0
    pmask = (periods <= max_p) & (periods >= 2)
    if np.any(pmask):
        axes[1, 1].semilogy(periods[pmask], pwr[pmask], 'b-', linewidth=0.8)
        axes[1, 1].semilogy(periods[pmask], r['conf_95'][pmask], 'g--',
                            linewidth=1, alpha=0.6, label='95%')
        axes[1, 1].semilogy(periods[pmask], r['conf_99'][pmask], 'r--',
                            linewidth=1, alpha=0.6, label='99%')
        axes[1, 1].set_xlim(2, max_p)
    axes[1, 1].set_title(f'Segment {i+1}: Power vs Period')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\n" + "="*75)
print("Analysis complete.")
print("="*75)
