"""
Persistent browser rendering for repeated, fast frame captures.

``convert_animation`` starts a web server and a Chromium instance, renders
once, and tears everything down — ~1 s of fixed cost per render. When the
same animation must be rendered many times (parameter optimization, sampling
sweeps), that cost dominates. This module removes it:

- ``RenderSession`` keeps one uvicorn server + one Playwright page alive for
  the lifetime of the session. Repeated captures reuse the loaded page — no
  server start, browser launch, page load, or font wait per render.
- Frame capture is batched: ``seekAndAppendToDomUsingTimes`` (convert.js)
  seeks the GSAP timeline N times and stacks N fixed-size SVG snapshots
  vertically in the DOM, so N frames are captured with a **single**
  full-page screenshot and sliced into numpy arrays — instead of N
  seek + screenshot round-trips.
- ``BrowserPool`` holds several sessions for parallel rendering from
  multiple threads.

All public methods are synchronous and thread-safe: each session runs its own
asyncio event loop on a daemon thread and callers block on
``run_coroutine_threadsafe`` futures.

Typical cost on an M-series laptop: ~1 s one-time startup per session, then
~3 ms per frame for batched captures (vs ~1 s per render with
``convert_animation``).
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import uvicorn
from PIL import Image
from playwright.async_api import async_playwright

from mover.converter.mover_converter import (
    _get_bound_port,
    _wait_for_server_start,
    setup_fastapi_app,
)

## Chromium rejects/clips capture surfaces beyond ~16384 px on a side.
_MAX_SCREENSHOT_HEIGHT_PX = 16_000


class RenderSession:
    """
    One persistent (server + browser page) pair serving a single HTML file.

    Usage::

        session = RenderSession("animation.html")
        session.start()
        frames, duration = session.capture_frames_at_fractions(np.linspace(0, 1, 30))
        session.close()

    ``evaluate()`` exposes the page for custom JavaScript (e.g. mutating
    animation parameters before a capture); the page's global scope persists
    across calls.

    Parameters
    ----------
    html_file : str | Path
        Animation HTML file. Served at ``/`` with the converter assets
        (convert.js etc.) mounted alongside, exactly like ``convert_animation``.
    output_dir : str | Path | None
        Where the FastAPI JSON endpoints write their files (only relevant if
        the page posts conversion data). Defaults to the HTML file's directory.
    frame_size : int
        Side length in pixels of captured square frames.
    max_frames_per_screenshot : int
        Upper bound on frames captured per screenshot. The effective bound is
        additionally capped so a screenshot never exceeds ~16k px in height
        (Chromium clips taller capture surfaces).
    """

    def __init__(self, html_file: str | Path, output_dir: str | Path | None = None,
                 frame_size: int = 128, max_frames_per_screenshot: int = 100):
        self.html_file = Path(html_file)
        self.output_dir = str(output_dir) if output_dir is not None else None
        self.frame_size = frame_size
        self.max_frames_per_screenshot = max_frames_per_screenshot
        self.port: int | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None
        self._pw = None
        self._browser = None
        self._page = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self, timeout: float = 60.0) -> "RenderSession":
        future = self.start_future()
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise
        return self

    def start_future(self):
        """Begin startup without blocking; returns a concurrent Future.

        Lets ``BrowserPool`` start many sessions in parallel.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        return asyncio.run_coroutine_threadsafe(self._start_async(), self._loop)

    async def _start_async(self) -> None:
        app = setup_fastapi_app(
            html_file=str(self.html_file),
            html_dir=str(self.html_file.parent),
            base_name=self.html_file.stem,
            output_dir=self.output_dir,
        )
        ## port=0: the OS picks a free port, so any number of sessions can
        ## coexist without port bookkeeping.
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.ensure_future(self._server.serve())
        await _wait_for_server_start(self._server, self._server_task)
        self.port = _get_bound_port(self._server)

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        self._page = await self._browser.new_page(device_scale_factor=1)
        await self._page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")
        await self._page.evaluate("document.fonts.ready")

    def close(self, timeout: float = 30.0) -> None:
        if self._loop is None:
            return
        self._call(self._close_async(), timeout=timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop = None

    async def _close_async(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        if self._server is not None:
            self._server.should_exit = True
            await self._server_task

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(self, expression: str, arg=None):
        """Run JavaScript in the page (same semantics as Playwright's
        ``page.evaluate``). Thread-safe; blocks until the result is available."""
        return self._call(self._page.evaluate(expression, arg))

    def get_animation_info(self, fps: int = 60) -> dict:
        """Return ``{animDuration, fps, steps}`` for the page's timeline."""
        return self.evaluate("fps => getAnimationInfo(fps)", fps)

    def capture_frames_at_times(self, seek_times: list[float]) -> list[np.ndarray]:
        """Render the animation at each seek time (seconds) and return one
        float32 RGBA array in [0, 1] of shape (frame_size, frame_size, 4) per
        time, captured in batches of ``max_frames_per_screenshot`` per
        screenshot."""
        return self._call(self._capture_async([float(t) for t in seek_times]))

    def capture_frames_at_fractions(self, fractions: list[float]) -> tuple[list[np.ndarray], float]:
        """Like ``capture_frames_at_times`` but positions are fractions in
        [0, 1] of the timeline's **current** duration. Returns
        ``(frames, duration_seconds)`` — useful when custom JavaScript may have
        changed the animation length between captures."""
        duration = float(self.get_animation_info()["animDuration"])
        frames = self.capture_frames_at_times([f * duration for f in fractions])
        return frames, duration

    # ── Internals ────────────────────────────────────────────────────────────

    def _call(self, coro, timeout: float = 90.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            ## result() timing out does not stop the coroutine — cancel it so a
            ## zombie capture can't fire resetSeekAndAppend into a later render.
            future.cancel()
            raise

    async def _capture_async(self, seek_times: list[float]) -> list[np.ndarray]:
        size = self.frame_size
        ## Chromium clips capture surfaces beyond ~16k px, which would make the
        ## slicing below silently return empty/partial frames — bound each
        ## screenshot's height in pixels, not just in frames.
        per_shot = min(self.max_frames_per_screenshot, max(1, _MAX_SCREENSHOT_HEIGHT_PX // size))
        frames: list[np.ndarray] = []
        for start in range(0, len(seek_times), per_shot):
            chunk = seek_times[start:start + per_shot]
            await self._page.set_viewport_size({"width": size, "height": size * len(chunk)})
            try:
                await self._page.evaluate(
                    "([times, size]) => seekAndAppendToDomUsingTimes(times, size)", [chunk, size]
                )
                ## Two rAFs guarantee the appended frames have been painted.
                await self._page.evaluate(
                    "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
                )
                png_bytes = await self._page.screenshot(type="png", full_page=True)
            finally:
                ## Runs even after a mid-append failure (e.g. a broken timeline
                ## throwing in seek) — leftover wrappers and a hidden source SVG
                ## would corrupt every subsequent capture on this page.
                await self._page.evaluate("() => resetSeekAndAppend()")
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            stack = np.asarray(img, dtype=np.float32) / 255.0
            if stack.shape[0] < size * len(chunk) or stack.shape[1] < size:
                raise RuntimeError(
                    f"Screenshot is {stack.shape[1]}x{stack.shape[0]} px but {len(chunk)} "
                    f"frames of {size} px need {size}x{size * len(chunk)} — refusing to "
                    f"slice a clipped capture into corrupted frames"
                )
            for i in range(len(chunk)):
                frames.append(stack[i * size:(i + 1) * size, :size])
        return frames


class BrowserPool:
    """
    N ``RenderSession`` workers with queue-based checkout for parallel
    rendering from multiple threads.

    Usage::

        pool = BrowserPool("animation.html", n_workers=4)
        with pool.acquire() as session:
            frames, duration = session.capture_frames_at_fractions(fractions)
        pool.shutdown()

    ``acquire()`` blocks until a session is free and returns it to the pool
    when the ``with`` block exits, so arbitrary sequences of ``evaluate()`` and
    capture calls run on one session without interleaving from other threads.
    """

    def __init__(self, html_file: str | Path, n_workers: int,
                 output_dir: str | Path | None = None, frame_size: int = 128,
                 start_timeout: float = 60.0):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self._sessions = [
            RenderSession(html_file, output_dir=output_dir, frame_size=frame_size)
            for _ in range(n_workers)
        ]
        self._available: queue.Queue[RenderSession] = queue.Queue()
        futures = [s.start_future() for s in self._sessions]
        for session, future in zip(self._sessions, futures):
            future.result(timeout=start_timeout)
            self._available.put(session)

    @property
    def sessions(self) -> tuple[RenderSession, ...]:
        """All sessions, e.g. for one-time per-session setup (warm-up
        evaluations) right after construction. Do NOT use for rendering while
        other threads hold checkouts — go through ``acquire()`` instead."""
        return tuple(self._sessions)

    @contextmanager
    def acquire(self) -> Iterator[RenderSession]:
        session = self._available.get()
        try:
            yield session
        finally:
            self._available.put(session)

    def shutdown(self) -> None:
        for session in self._sessions:
            session.close()
