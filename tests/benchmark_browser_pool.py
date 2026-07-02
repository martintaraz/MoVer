"""
Benchmark: what do session reuse and batched capture actually buy?

Compares four ways of rendering the same N frames of an animation:

  A. cold + per-frame   — fresh server/browser per render, one screenshot per
                          frame (approximates the pre-pool flow; the real
                          convert_animation path also extracts JSON, so it
                          was slower still)
  B. cold + batched     — fresh server/browser per render, one screenshot per
                          render (isolates remaining startup cost)
  C. warm + per-frame   — persistent RenderSession, one screenshot per frame
                          (isolates the page-reuse gain)
  D. warm + batched     — persistent RenderSession, one screenshot per render
                          (the browser_pool fast path)

Run with:
    python tests/benchmark_browser_pool.py [html_file] [--n-frames 30] [--n-renders 10] [--n-cold-renders 3] [--json out.json]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mover.converter.browser_pool import RenderSession

DEFAULT_HTML = Path(__file__).parent / "assets" / "test_animation.html"


def timed(fn, n: int) -> list[float]:
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def summarize(label: str, samples: list[float], n_frames: int) -> dict:
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"  {label:<18} {mean * 1000:9.1f} ms/render ± {std * 1000:6.1f}   "
          f"({mean * 1000 / n_frames:6.2f} ms/frame, n={len(samples)})")
    return {"label": label, "mean_s": mean, "std_s": std, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark persistent + batched rendering against the cold per-frame status quo.")
    parser.add_argument("html_file", type=str, nargs="?", default=str(DEFAULT_HTML), help="Animation HTML file to render")
    parser.add_argument("--n-frames", type=int, default=30, help="Frames per render (default: 30)")
    parser.add_argument("--n-renders", type=int, default=10, help="Renders per warm measurement (default: 10)")
    parser.add_argument("--n-cold-renders", type=int, default=3, help="Renders per cold measurement; each pays full startup (default: 3)")
    parser.add_argument("--frame-size", type=int, default=128, help="Square frame side length in px (default: 128)")
    parser.add_argument("--json", type=str, default=None, help="Also write results to this JSON file")
    args = parser.parse_args()

    fractions = list(np.linspace(0.0, 1.0, args.n_frames))

    print(f"Benchmarking {args.html_file}")
    print(f"  {args.n_frames} frames/render @ {args.frame_size}px\n")

    ## Warm session, also used to resolve seek times once for all modes.
    start = time.perf_counter()
    session = RenderSession(args.html_file, frame_size=args.frame_size).start()
    startup_s = time.perf_counter() - start
    print(f"  session startup    {startup_s * 1000:9.1f} ms (one-time)\n")
    times = [f * float(session.get_animation_info()["animDuration"]) for f in fractions]

    def cold(batched: bool):
        s = RenderSession(args.html_file, frame_size=args.frame_size).start()
        try:
            if batched:
                s.capture_frames_at_times(times)
            else:
                for t in times:
                    s.capture_frames_at_times([t])
        finally:
            s.close()

    results = {
        "html_file": str(args.html_file), "n_frames": args.n_frames,
        "frame_size": args.frame_size, "startup_s": startup_s,
        "modes": [
            summarize("cold, per-frame", timed(lambda: cold(batched=False), args.n_cold_renders), args.n_frames),
            summarize("cold, batched", timed(lambda: cold(batched=True), args.n_cold_renders), args.n_frames),
            summarize("warm, per-frame", timed(lambda: [session.capture_frames_at_times([t]) for t in times], args.n_renders), args.n_frames),
            summarize("warm, batched", timed(lambda: session.capture_frames_at_times(times), args.n_renders), args.n_frames),
        ],
    }
    session.close()

    baseline = results["modes"][0]["mean_s"]
    fast = results["modes"][3]["mean_s"]
    print(f"\n  speedup (warm+batched vs cold+per-frame): {baseline / fast:.1f}x")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"  results written to {args.json}")


if __name__ == "__main__":
    main()
