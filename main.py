"""
main.py — SCF experiment on Deep10m (deep-image-96-angular.hdf5).

Usage:
    python main.py                                          # H100 defaults
    python main.py --data /path/to/deep-image-96-angular.hdf5
    python main.py --backend numpy --max_corpus 100000 --num_queries 500
    python main.py --modes 1bit_sign 2bit_bf16
"""

import argparse, os
from scf_evaluator import (
    MODES, load_hdf5_dataset, plot_results,
    print_operating_points, print_results_table,
    run_experiments, save_results, threshold_range_from_fraction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "/home/cmehta/Experiment/data/deep-image-96-angular.hdf5"

K          = 32
BACKEND    = "cupy"
BATCH_SIZE = 256

THRESHOLD_START_FRAC = 0.40
THRESHOLD_STOP_FRAC  = 0.99
THRESHOLD_STEP_FRAC  = 0.005
RECALL_STOP          = 0.75    # must exceed max operating-point target (0.95)

MANUAL_BIT_WEIGHTS = {
    "2bit": [0.99, 0.01],
    "4bit": [0.90, 0.06, 0.03, 0.01],
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=DEFAULT_DATA_PATH)
    parser.add_argument("--backend",     choices=["cupy","numpy"], default=BACKEND)
    parser.add_argument("--max_corpus",  type=int, default=9_990_000)
    parser.add_argument("--num_queries", type=int, default=10_000)
    parser.add_argument("--modes",       nargs="+", choices=list(MODES.keys()),
                        default=list(MODES.keys()))
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"SCF Filtering — Deep10m-96")
    print(f"backend={args.backend}  K={K}  batch={BATCH_SIZE}")
    print(f"modes={args.modes}")
    print(f"corpus≤{args.max_corpus:,}  queries≤{args.num_queries:,}")
    print(f"threshold sweep: [{THRESHOLD_START_FRAC}, {THRESHOLD_STOP_FRAC}] "
          f"step={THRESHOLD_STEP_FRAC}  recall_stop={RECALL_STOP}")
    print("=" * 60)

    print(f"\nLoading {args.data} ...")
    corpus, queries = load_hdf5_dataset(
        args.data,
        max_corpus=args.max_corpus,
        num_queries=args.num_queries,
    )
    dim = corpus.shape[1]
    print(f"  corpus {corpus.shape}  queries {queries.shape}  D={dim}")

    experiments = [
        {
            "name":    MODES[m]["label"],
            "mode":    m,
            "backend": args.backend,
            "threshold_range": threshold_range_from_fraction(
                m, dim,
                THRESHOLD_START_FRAC, THRESHOLD_STOP_FRAC, THRESHOLD_STEP_FRAC,
            ),
            "batch_size":  BATCH_SIZE,
            "recall_stop": RECALL_STOP,
        }
        for m in args.modes
    ]

    results = run_experiments(
        corpus, queries, experiments, k=K,
        manual_weights_map=MANUAL_BIT_WEIGHTS,
    )

    save_results(results, "results/deep10m_results.csv")
    print_results_table(results, K)
    print("\nOperating points:")
    print_operating_points(results, K)
    plot_results(results, K, "results/deep10m_curves.png",
                 corpus_size=corpus.shape[0], num_queries=queries.shape[0])
    print("\nDone.")

if __name__ == "__main__":
    main()