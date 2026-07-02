"""
Unit tests for mover.converter.browser_pool.

Run with:
    python tests/test_browser_pool.py

Uses tests/assets/test_animation.html, which loads the vendored GSAP from the
converter assets mount — no network access needed (CI-friendly). One
RenderSession is shared across tests; expect a total runtime of ~10-30 s.
"""

import threading
import unittest
from pathlib import Path

import numpy as np

from mover.converter.browser_pool import BrowserPool, RenderSession

TEST_HTML = Path(__file__).parent / "assets" / "test_animation.html"
ANIMATION_DURATION = 2.0
FRAME_SIZE = 128

_session = None


def setUpModule():
    global _session
    _session = RenderSession(TEST_HTML, frame_size=FRAME_SIZE).start()


def tearDownModule():
    _session.close()


class TestRenderSession(unittest.TestCase):

    def test_animation_info(self):
        info = _session.get_animation_info(fps=60)
        self.assertAlmostEqual(info["animDuration"], ANIMATION_DURATION, places=3)
        self.assertEqual(info["fps"], 60)
        self.assertEqual(info["steps"], int(np.ceil(ANIMATION_DURATION * 60)))

    def test_capture_shapes_and_range(self):
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        frames = _session.capture_frames_at_times(times)
        self.assertEqual(len(frames), len(times))
        for frame in frames:
            self.assertEqual(frame.shape, (FRAME_SIZE, FRAME_SIZE, 4))
            self.assertEqual(frame.dtype, np.float32)
            self.assertGreaterEqual(frame.min(), 0.0)
            self.assertLessEqual(frame.max(), 1.0)

    def test_animation_actually_moves(self):
        first, last = _session.capture_frames_at_times([0.0, ANIMATION_DURATION])
        ## The ball travels 120/200 of the scene width; frames must differ a lot.
        self.assertGreater(np.abs(first - last).mean(), 0.005)

    def test_capture_at_fractions_returns_duration(self):
        frames, duration = _session.capture_frames_at_fractions([0.0, 0.5, 1.0])
        self.assertEqual(len(frames), 3)
        self.assertAlmostEqual(duration, ANIMATION_DURATION, places=3)

    def test_batched_equals_per_frame(self):
        """One N-frame screenshot must slice into the same pixels as N
        single-frame captures — guards slicing offset bugs (wrong row/column
        origin) and cross-frame contamination."""
        times = list(np.linspace(0.0, ANIMATION_DURATION, 8))
        batched = _session.capture_frames_at_times(times)
        individual = [_session.capture_frames_at_times([t])[0] for t in times]
        for i, (b, s) in enumerate(zip(batched, individual)):
            ## Identical DOM + deterministic renderer: allow only 1/255 noise.
            np.testing.assert_allclose(b, s, atol=1.5 / 255.0, err_msg=f"frame {i} (t={times[i]:.3f}s)")

    def test_chunked_equals_unchunked(self):
        times = list(np.linspace(0.0, ANIMATION_DURATION, 7))
        unchunked = _session.capture_frames_at_times(times)
        original = _session.max_frames_per_screenshot
        _session.max_frames_per_screenshot = 3
        try:
            chunked = _session.capture_frames_at_times(times)
        finally:
            _session.max_frames_per_screenshot = original
        self.assertEqual(len(chunked), len(times))
        for i, (c, u) in enumerate(zip(chunked, unchunked)):
            np.testing.assert_allclose(c, u, atol=1.5 / 255.0, err_msg=f"frame {i}")

    def test_frames_are_flush_with_svg_content(self):
        """Absolute alignment: a sliced frame must contain ONLY the SVG (its
        background color reaches all four corners) — no page background, prompt
        text, or button row bleeding in from layout offsets (html/body padding,
        body content before the SVG). Relative batched-vs-per-frame comparisons
        cannot catch constant offsets; this test can."""
        frame = _session.capture_frames_at_times([0.0])[0]
        expected = np.array([0.0, 200 / 255.0, 0.0, 1.0], dtype=np.float32)
        for name, corner in [("top-left", frame[0, 0]), ("top-right", frame[0, -1]),
                             ("bottom-left", frame[-1, 0]), ("bottom-right", frame[-1, -1])]:
            np.testing.assert_allclose(corner, expected, atol=0.02,
                                       err_msg=f"{name} corner is not SVG background")

    def test_capture_resets_page(self):
        _session.capture_frames_at_times([0.0, 1.0])
        leftovers = _session.evaluate("() => document.querySelectorAll('.mover-batch-frame').length")
        self.assertEqual(leftovers, 0)
        hidden_leftovers = _session.evaluate("() => document.querySelectorAll('[data-mover-batch-display]').length")
        self.assertEqual(hidden_leftovers, 0)
        svg_display = _session.evaluate(
            "() => getComputedStyle(document.querySelector('body > svg')).display"
        )
        self.assertNotEqual(svg_display, "none")
        ## Elements with their own inline display:none must stay hidden after reset.
        sys_msg_display = _session.evaluate(
            "() => getComputedStyle(document.querySelector('#sys-msg-path')).display"
        )
        self.assertEqual(sys_msg_display, "none")
        prompt_display = _session.evaluate(
            "() => getComputedStyle(document.querySelector('#prompt')).display"
        )
        self.assertNotEqual(prompt_display, "none")

    def test_failed_capture_cleans_up(self):
        """A timeline that throws mid-append must not leave frame wrappers or a
        hidden source SVG behind — they would corrupt every subsequent capture
        on this session."""
        _session.evaluate("""() => {
            window._origSeek = tl_to_use.seek.bind(tl_to_use);
            let calls = 0;
            tl_to_use.seek = (...args) => {
                if (++calls >= 2) throw new Error('boom');
                return window._origSeek(...args);
            };
        }""")
        try:
            with self.assertRaises(Exception):
                _session.capture_frames_at_times([0.0, 0.5, 1.0])
        finally:
            _session.evaluate("() => { tl_to_use.seek = window._origSeek; }")
        leftovers = _session.evaluate("() => document.querySelectorAll('.mover-batch-frame').length")
        self.assertEqual(leftovers, 0)
        frames = _session.capture_frames_at_times([0.0, ANIMATION_DURATION])
        self.assertEqual(len(frames), 2)
        self.assertGreater(np.abs(frames[0] - frames[1]).mean(), 0.005)

    def test_evaluate_persists_page_globals(self):
        """evaluate() must see and mutate the page's global scope across calls
        — external tooling relies on this to rebuild timelines in place."""
        _session.evaluate("() => { window._pool_test_marker = 42; }")
        self.assertEqual(_session.evaluate("() => window._pool_test_marker"), 42)
        self.assertEqual(_session.evaluate("() => typeof tl_to_use"), "object")


class TestBrowserPool(unittest.TestCase):

    def test_parallel_renders_from_threads(self):
        pool = BrowserPool(TEST_HTML, n_workers=2, frame_size=FRAME_SIZE)
        try:
            fractions = list(np.linspace(0.0, 1.0, 5))
            with pool.acquire() as session:
                reference, ref_duration = session.capture_frames_at_fractions(fractions)

            results = [None] * 4
            def render(i):
                with pool.acquire() as session:
                    results[i] = session.capture_frames_at_fractions(fractions)
            threads = [threading.Thread(target=render, args=(i,)) for i in range(len(results))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=90)

            for i, result in enumerate(results):
                self.assertIsNotNone(result, f"render {i} did not finish")
                frames, duration = result
                self.assertEqual(len(frames), len(fractions))
                self.assertAlmostEqual(duration, ref_duration, places=3)
                for j, frame in enumerate(frames):
                    np.testing.assert_allclose(
                        frame, reference[j], atol=1.5 / 255.0,
                        err_msg=f"render {i}, frame {j} differs from reference",
                    )
        finally:
            pool.shutdown()

    def test_rejects_invalid_worker_count(self):
        with self.assertRaises(ValueError):
            BrowserPool(TEST_HTML, n_workers=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
