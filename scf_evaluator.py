"""
scf_evaluator.py — SCF filtering via BF16 bit extraction (1/2/4-bit modes).
"""

import csv, os, time
import numpy as np


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def get_xp(backend):
    if backend == "numpy": return np
    if backend == "cupy":
        import cupy as cp; return cp
    raise ValueError(f"Unknown backend: {backend!r}")

def to_np(arr):
    if isinstance(arr, np.ndarray): return arr
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray): return cp.asnumpy(arr)
    except ImportError: pass
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_hdf5_dataset(filepath, max_corpus=9_990_000, num_queries=10_000):
    import h5py
    with h5py.File(filepath, 'r') as f:
        corpus  = f['train'][:max_corpus].astype(np.float32)
        queries = f['test'][:num_queries].astype(np.float32)
    return corpus, queries


# ---------------------------------------------------------------------------
# Normalisation + BF16 quantisation
# ---------------------------------------------------------------------------

def normalize(vecs, xp):
    return vecs / xp.maximum(xp.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)

def to_bf16(vecs_np):
    """float32 → bf16 stored as uint16 (top 16 bits of float32 IEEE repr)."""
    u32 = vecs_np.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Encoding
# BF16 layout: [sign(b15) | exp(b14:b8) | mantissa(b7:b0)]
# sign plane flipped so 1=positive; exponent planes left as-is.
# ---------------------------------------------------------------------------

def _extract_bit_planes(bf16_u16, bit_positions):
    planes = [(bf16_u16 >> b) & np.uint16(1) for b in bit_positions]
    return np.concatenate(planes, axis=1).astype(np.int8)

def encode_1bit(vecs_np):
    codes = _extract_bit_planes(to_bf16(vecs_np), [15])
    return 1 - codes                                      # flip: 1=positive

def encode_2bit(vecs_np):
    codes = _extract_bit_planes(to_bf16(vecs_np), [15, 14])
    codes[:, :vecs_np.shape[1]] = 1 - codes[:, :vecs_np.shape[1]]  # flip sign plane only
    return codes

def encode_4bit(vecs_np):
    codes = _extract_bit_planes(to_bf16(vecs_np), [15, 14, 13, 12])
    codes[:, :vecs_np.shape[1]] = 1 - codes[:, :vecs_np.shape[1]]  # flip sign plane only
    return codes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def concordance_scores(corpus_codes, query_codes_batch, xp):
    """Unweighted: score = #matching bits across all planes."""
    B   = corpus_codes.shape[1]
    c1  = corpus_codes.sum(axis=1, dtype=xp.int32)
    q1  = query_codes_batch.sum(axis=1, keepdims=True, dtype=xp.int32)
    AND = query_codes_batch.astype(xp.int32) @ corpus_codes.T.astype(xp.int32)
    return (B - q1 - c1 + 2 * AND).astype(xp.float32)

def weighted_concordance_scores(corpus_codes, query_codes_batch, xp,
                                weights_per_plane, ceiling):
    """
    Hard-gate weighted concordance (DReX formulation):
        score(q,c) = sum_i  sign_match[i] * sum_k( w[k] * bit_match_k[i] )
    Sign mismatch at dim i zeroes that dim's contribution entirely.
    For each plane k, counts dims where sign AND plane-k both agree,
    decomposed into 4 matmuls on {0,1} vectors.
    """
    n_planes = len(weights_per_plane)
    dim      = corpus_codes.shape[1] // n_planes

    def planes(codes):
        return [codes[:, k*dim:(k+1)*dim].astype(xp.float32) for k in range(n_planes)]

    cc_p = planes(corpus_codes)
    qc_p = planes(query_codes_batch)
    w    = xp.asarray(weights_per_plane, dtype=xp.float32)
    acc  = xp.zeros((query_codes_batch.shape[0], corpus_codes.shape[0]), dtype=xp.float32)

    sq, sc   = qc_p[0], cc_p[0]     # sign planes
    sq_, sc_ = 1.0 - sq, 1.0 - sc

    for k in range(n_planes):
        if weights_per_plane[k] == 0.0: continue
        mq, mc   = qc_p[k], cc_p[k]
        mq_, mc_ = 1.0 - mq, 1.0 - mc
        # dims where sign agrees AND plane-k agrees
        both = (sq  * mq ) @ (sc  * mc ).T \
             + (sq  * mq_) @ (sc  * mc_).T \
             + (sq_ * mq ) @ (sc_ * mc ).T \
             + (sq_ * mq_) @ (sc_ * mc_).T
        acc += w[k] * both

    return (acc * (ceiling / (float(w.sum()) * dim))).astype(xp.float32)

def make_manual_score_fn(weights, ceiling):
    return lambda cc, qc, xp: weighted_concordance_scores(cc, qc, xp, weights, ceiling)


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

MODES = {
    "1bit_sign": {
        "label": "1-bit sign", "bits_per_dim": 1,
        "encode_fn": encode_1bit, "manual_weights": None,
    },
    "2bit_bf16": {
        "label": "2-bit BF16", "bits_per_dim": 2,
        "encode_fn": encode_2bit, "manual_weights": None,
    },
    "4bit_bf16": {
        "label": "4-bit BF16", "bits_per_dim": 4,
        "encode_fn": encode_4bit, "manual_weights": None,
    },
    "2bit_manual": {
        "label": "2-bit manual-weighted", "bits_per_dim": 2,
        "encode_fn": encode_2bit, "manual_weights": "2bit",
    },
    "4bit_manual": {
        "label": "4-bit manual-weighted", "bits_per_dim": 4,
        "encode_fn": encode_4bit, "manual_weights": "4bit",
    },
}


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def score_ceiling(mode_name, dim):
    return MODES[mode_name]["bits_per_dim"] * dim

def threshold_range_from_fraction(mode_name, dim, start_frac, stop_frac, step_frac):
    ceil  = score_ceiling(mode_name, dim)
    start = int(round(start_frac * ceil))
    stop  = int(round(stop_frac  * ceil))
    step  = max(1, int(round(step_frac * ceil)))
    return range(start, stop + 1, step)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def compute_ground_truth(corpus_norm, queries_norm, k, xp, batch_size=256):
    gt = []
    for s in range(0, queries_norm.shape[0], batch_size):
        sims = queries_norm[s:s+batch_size] @ corpus_norm.T
        for row in to_np(xp.argpartition(sims, -k, axis=1)[:, -k:]):
            gt.append(set(row.tolist()))
    return gt


# ---------------------------------------------------------------------------
# Vectorized threshold sweep
# Eliminates Python loops over queries and thresholds.
# For a batch of Qb queries and T thresholds:
#   survivors : (T, Qb) — number of corpus vectors passing each threshold
#   recall    : (T, Qb) — recall@k for each query at each threshold
# ---------------------------------------------------------------------------

def _sweep_batch(scores, qvecs, corpus_norm, gt_batch, thresholds_arr, k, Nc, xp):
    """
    scores      : (Qb, Nc) float32
    qvecs       : (Qb, D)  float32  normalised
    thresholds_arr : (T,)  float32
    Returns: survivors (T, Qb), recalls (T, Qb), filter_ratios (T, Qb)
    """
    T, Qb = len(thresholds_arr), scores.shape[0]

    # (T, Qb, Nc) boolean mask — scores[q] >= threshold[t]
    # Avoid materialising full (T,Qb,Nc) on GPU: loop over T only (T is small ~110)
    survivors  = xp.zeros((T, Qb), dtype=xp.int32)
    recalls    = xp.zeros((T, Qb), dtype=xp.float32)

    for ti, thr in enumerate(thresholds_arr):
        mask = scores >= thr                              # (Qb, Nc) bool
        survivors[ti] = mask.sum(axis=1)                 # (Qb,)

        # Recall: for each query rerank survivors
        for qi in range(Qb):
            idx = xp.where(mask[qi])[0]
            n   = int(idx.shape[0])
            if n >= k:
                sims = corpus_norm[idx] @ qvecs[qi]
                top  = xp.argpartition(sims, -k)[-k:]
                pred = set(to_np(idx[top]).tolist())
            else:
                pred = set(to_np(idx).tolist()) if n > 0 else set()
            recalls[ti, qi] = len(pred & gt_batch[qi]) / k

    filter_ratios = Nc / xp.maximum(survivors, 1).astype(xp.float32)
    return to_np(survivors), to_np(recalls), to_np(filter_ratios)


def evaluate_filter_sweep(corpus, queries, k, threshold_range, mode_name,
                           manual_weights_map, backend="cupy",
                           batch_size=256, recall_stop=None):
    mode       = MODES[mode_name]
    xp         = get_xp(backend)
    thresholds = list(threshold_range)
    T          = len(thresholds)
    Nq, Nc     = queries.shape[0], corpus.shape[0]
    dim        = corpus.shape[1]
    ceiling    = score_ceiling(mode_name, dim)
    thr_arr    = np.array(thresholds, dtype=np.float32)

    cx = xp.asarray(corpus);  cn = normalize(cx, xp)
    qx = xp.asarray(queries); qn = normalize(qx, xp)
    gt = compute_ground_truth(cn, qn, k, xp, batch_size=batch_size)

    cc = xp.asarray(mode["encode_fn"](to_np(cx)))
    qc = xp.asarray(mode["encode_fn"](to_np(qx)))

    mw_key = mode["manual_weights"]
    if mw_key:
        raw_w         = manual_weights_map[mw_key]
        score_fn      = make_manual_score_fn(raw_w, ceiling)
        display_label = f"{mode['label']} {raw_w}"
    else:
        score_fn      = concordance_scores
        display_label = mode["label"]
        raw_w         = None

    # accumulators over queries
    rec_sum = np.zeros(T)
    fr_sum  = np.zeros(T)
    sv_sum  = np.zeros(T)
    active_T = T   # thresholds still in play (for early stop)

    for s in range(0, Nq, batch_size):
        e        = min(s + batch_size, Nq)
        scores   = score_fn(cc, qc[s:e], xp)             # (Qb, Nc)
        gt_batch = gt[s:e]

        sv, rc, fr = _sweep_batch(
            scores, qn[s:e], cn, gt_batch,
            thr_arr[:active_T], k, Nc, xp,
        )
        sv_sum[:active_T] += sv.sum(axis=1)
        rec_sum[:active_T] += rc.sum(axis=1)
        fr_sum[:active_T]  += fr.sum(axis=1)

    results = []
    for ti, thr in enumerate(thresholds):
        mr = rec_sum[ti] / Nq
        results.append({
            "mode": mode_name, "mode_label": display_label,
            "run_name": display_label, "backend": backend,
            "threshold": thr, "recall": mr,
            "filter_ratio": fr_sum[ti] / Nq,
            "avg_survivors": sv_sum[ti] / Nq,
            "bit_weights": str(raw_w) if raw_w else "",
        })
        if recall_stop is not None and mr < recall_stop:
            print(f"    early-stop thr={thr} recall={mr:.3f}"); break
    return results


def run_experiments(corpus, queries, experiments, k, manual_weights_map):
    all_results, t0 = [], time.time()
    for exp in experiments:
        print(f"\n  [{exp['mode']}]  {exp['name']}")
        t1   = time.time()
        rows = evaluate_filter_sweep(
            corpus, queries, k=k,
            threshold_range=exp["threshold_range"],
            mode_name=exp["mode"],
            manual_weights_map=manual_weights_map,
            backend=exp["backend"],
            batch_size=exp["batch_size"],
            recall_stop=exp.get("recall_stop"),
        )
        print(f"    → {len(rows)} points  {time.time()-t1:.1f}s")
        all_results.extend(rows)
    m, s = divmod(int(time.time() - t0), 60)
    print(f"\n  total: {m}m {s}s")
    return all_results


# ---------------------------------------------------------------------------
# Results + plotting
# ---------------------------------------------------------------------------

def group_by_run(results):
    g = {}
    for r in results:
        g.setdefault(r["run_name"], []).append(r)
    for key in g:
        seen, out = set(), []
        for r in g[key]:
            if r["threshold"] not in seen:
                seen.add(r["threshold"]); out.append(r)
        g[key] = out
    return g

def closest_recall(rows, target):
    return min(rows, key=lambda r: abs(r["recall"] - target))

def print_results_table(results, k):
    print(f"\n{'Run':<38} {'TH':>7} {'Recall@'+str(k):>10} {'FR':>10} {'Survivors':>12}")
    print("-" * 82)
    for r in results:
        print(f"{r['run_name']:<38} {r['threshold']:>7.1f} {r['recall']:>10.4f} "
              f"{r['filter_ratio']:>10.1f} {r['avg_survivors']:>12.0f}")

def print_operating_points(results, k, targets=(0.95, 0.90, 0.80)):
    for run, rows in group_by_run(results).items():
        print(f"\n  {run}")
        for t in targets:
            r = closest_recall(rows, t)
            print(f"    recall≈{t:.2f}  TH={r['threshold']:.1f}  "
                  f"FR={r['filter_ratio']:.1f}x  actual={r['recall']:.3f}")

def save_results(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["run_name","mode","mode_label","backend","threshold",
              "recall","filter_ratio","avg_survivors","bit_weights"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    print(f"  saved → {path}")

_STYLE = {
    "1bit_sign":   {"color": "steelblue",   "ls": "-",  "marker": "o", "lw": 2.5},
    "2bit_bf16":   {"color": "darkorange",  "ls": "--", "marker": "s", "lw": 2},
    "4bit_bf16":   {"color": "forestgreen", "ls": "--", "marker": "^", "lw": 2},
    "2bit_manual": {"color": "darkorange",  "ls": "-",  "marker": "s", "lw": 2},
    "4bit_manual": {"color": "forestgreen", "ls": "-",  "marker": "^", "lw": 2},
}

def plot_results(results, k, path, corpus_size=None, num_queries=None):
    import math, matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(9, 6))
    for run, rows in group_by_run(results).items():
        s    = _STYLE.get(rows[0]["mode"], {"color":"gray","ls":"-","marker":"o","lw":2})
        rows = sorted(rows, key=lambda r: r["recall"])
        recalls = [r["recall"] for r in rows]
        frs     = [r["filter_ratio"] for r in rows]
        ax.plot(recalls, frs, ls=s["ls"], color=s["color"],
                marker=s["marker"], lw=s["lw"], markersize=4, label=run)

    all_recalls = [r["recall"] for r in results]
    mx = max(r["filter_ratio"] for r in results)
    ax.set_yscale("log")
    ax.set_ylim(top=10**math.ceil(math.log10(mx)))
    ax.set_xlim(max(0.75, min(all_recalls) - 0.02), 1.01)
    ax.set_xlabel(f"Recall@{k}", fontsize=11)
    ax.set_ylabel("Filter Ratio", fontsize=11)
    title = f"SCF | Deep10m-96 | Recall@{k}"
    if corpus_size and num_queries:
        title += f"\ncorpus={corpus_size:,}  queries={num_queries:,}"
    ax.set_title(title, fontsize=11)
    ax.axvline(x=0.95, color="gray", ls="--", alpha=0.5)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  plot → {path}")