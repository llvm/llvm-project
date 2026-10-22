"""Progress bar and measurement loop: runs lit, collects samples, writes summary.json."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from platform import machine, release

from .config import BenchmarkConfig
from .platforms.base import Platform
from .util import Stats, SYSTEM, _fmt_duration, format_bytes, log, summarize_samples


class ProgressBar:
    """A dependency-free progress indicator.

    On a TTY it renders a single live line (bar, count, elapsed/ETA, last
    sample) updated in place. Off a TTY it falls back to one log line per
    update so CI logs stay readable. Output goes to the given stream
    (stderr by default) to avoid colliding with stdout logging.
    """

    WIDTH = 24

    def __init__(self, total: int, label: str, stream=None):
        self.total = total
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._full, self._empty = ("#", "-")
        self.start = time.perf_counter()

    def update(self, done: int, last: float | None = None) -> None:
        if not self.enabled:
            tail = f": {last:.3f}s" if last is not None else ""
            log.info(f"{self.label} {done}/{self.total}{tail}")
            return
        frac = done / self.total if self.total else 1.0
        filled = round(self.WIDTH * frac)
        bar = self._full * filled + self._empty * (self.WIDTH - filled)
        elapsed = time.perf_counter() - self.start
        eta = elapsed / done * (self.total - done) if done else 0.0
        tail = f"  last {last:.3f}s" if last is not None else ""
        self.stream.write(
            f"\r{self.label} [{bar}] {done}/{self.total}  "
            f"{_fmt_duration(elapsed)}<{_fmt_duration(eta)}{tail}"
        )
        self.stream.flush()

    def finish(self) -> None:
        if self.enabled:
            self.stream.write("\r\033[K")
            self.stream.flush()


class BenchmarkRunner:
    """Runs the measurement loop and writes results."""

    def __init__(self, config: BenchmarkConfig, platform: Platform):
        self.config = config
        self.platform = platform

    def run(self) -> None:
        lit_cmd = self.platform.lit_command()
        with self.platform:
            self.platform.note_pinning()
            walls, rss = self._measure(lit_cmd)
        wall_stats = summarize_samples(walls) if walls else None
        rss_stats = summarize_samples(rss) if rss else None
        summary = self._build_summary(lit_cmd, walls, wall_stats, rss, rss_stats)
        self.config.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
        self._log_results(wall_stats, rss_stats)

    def _measure(self, lit_cmd: list[str]) -> tuple[list[float], list[int]]:
        cfg = self.config
        log.info(
            f"=== Measuring: {cfg.test_path} (-j{cfg.workers}, "
            f"{cfg.warmup} warmup + {cfg.runs} runs) ==="
        )
        # Human-readable echo of the exact lit invocation; the unambiguous
        # argv list is recorded under "lit_command" in summary.json.
        log.info(f"=== Command: {' '.join(lit_cmd)} ===")
        if cfg.warmup:
            self._run_phase(lit_cmd, cfg.warmup, "Warmup", collect=False)
        return self._run_phase(lit_cmd, cfg.runs, "Run", collect=True)

    def _run_phase(
        self, lit_cmd: list[str], count: int, label: str, collect: bool
    ) -> tuple[list[float], list[int]]:
        bar = ProgressBar(count, label)
        walls: list[float] = []
        rss: list[int] = []
        rss_supported = True
        try:
            for i in range(1, count + 1):
                sample = self.platform.measure_run(lit_cmd, f"{label} {i}")
                if collect:
                    walls.append(sample.wall_sec)
                    if sample.peak_rss_bytes is None:
                        rss_supported = False
                    elif rss_supported:
                        rss.append(sample.peak_rss_bytes)
                bar.update(i, sample.wall_sec)
        finally:
            bar.finish()
        if not rss_supported:
            # Drop any partial RSS samples so the stats reflect a clean subset.
            rss = []
        return walls, rss

    def _build_summary(
        self,
        lit_cmd: list[str],
        walls: list[float],
        wall_stats: Stats | None,
        rss: list[int],
        rss_stats: Stats | None,
    ) -> dict[str, object]:
        cfg = self.config
        wall_clock: dict[str, object] | None = None
        if wall_stats is not None:
            wall_clock = dict(wall_stats)
            wall_clock["samples_sec"] = walls
        peak_rss: dict[str, object] | None = None
        if rss_stats is not None:
            peak_rss = {"semantics": self.platform.RSS_SEMANTICS}
            peak_rss.update(rss_stats)
            peak_rss["samples_bytes"] = rss
        return {
            "schema_version": 2,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": {
                "system": SYSTEM,
                "release": release(),
                "machine": machine(),
                "cpu_count": os.cpu_count(),
            },
            "benchmark_python": sys.version.split()[0],
            "lit": str(cfg.lit),
            "test_path": str(cfg.test_path),
            "lit_command": lit_cmd,
            "workers": cfg.workers,
            "warmup": cfg.warmup,
            "runs": cfg.runs,
            "benchmark_cpus": list(cfg.benchmark_cpus),
            "cpu_pinning": self.platform.PINS_CPUS,
            "env_setup_skipped": cfg.skip_env,
            "isolation_applied": self.platform.applied,
            "wall_clock_sec": wall_clock,
            "peak_rss": peak_rss,
        }

    def _log_results(
        self,
        wall_stats: Stats | None,
        rss_stats: Stats | None,
    ) -> None:
        log.info("=== Results ===")
        if wall_stats is not None:
            stddev = wall_stats["stddev"]
            sd = f", stddev {stddev:.3f}" if stddev is not None else ""
            log.info(
                "Wall clock: median {median:.3f} s, mean {mean:.3f} s "
                "(min {min:.3f}, max {max:.3f}{sd}, n={n})".format(sd=sd, **wall_stats)
            )
        if rss_stats is not None:
            log.info(
                "Peak RSS:   median {median} (min {min}, max {max}, n={n})".format(
                    median=format_bytes(rss_stats["median"]),
                    min=format_bytes(rss_stats["min"]),
                    max=format_bytes(rss_stats["max"]),
                    n=rss_stats["n"],
                )
            )
        log.info(f"Summary: {self.config.summary_json}")
