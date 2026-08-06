"""Base per-OS strategy: lifecycle, command construction, and measurement.

The base class doubles as the fallback for unrecognized POSIX systems:
no tuning, no pinning, wait4-based timing and peak-RSS measurement.
Subclasses override only the hooks their OS supports, so the environment
and the spawn can never be mismatched.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable

from ..config import BenchmarkConfig, RunSample
from ..util import BenchmarkError, SYSTEM, _exit_code_from_wait_status, log


class Platform:
    """Per-OS strategy: environment tuning, CPU pinning, and per-run measurement.

    The base class doubles as the fallback for unrecognized POSIX systems:
    no tuning, no pinning, wait4-based timing and peak-RSS measurement.
    Subclasses override only the hooks their OS supports, so the environment
    and the spawn can never be mismatched.
    """

    PINS_CPUS = False
    RSS_SEMANTICS = (
        "max RSS over any single process in the benchmarked process tree "
        "(os.wait4 ru_maxrss)"
    )

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        # Record of isolation steps that actually applied,
        # for the reproducibility metadata in summary.json.
        self.applied: list[str] = []

    def setup(self) -> None:
        pass

    def restore(self) -> None:
        pass

    def __enter__(self) -> Platform:
        if self.config.skip_env:
            log.info("=== Environment setup skipped (--skip-env-setup) ===")
            return self
        try:
            self.setup()
        except BaseException:
            self._restore_safely()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        if not self.config.skip_env:
            self._restore_safely()

    def _restore_safely(self) -> None:
        try:
            self.restore()
        except Exception as err:
            log.error(
                f"Could not fully restore environment; manual cleanup may be required: {err}"
            )

    def lit_invocation(self) -> list[str]:
        return [str(self.config.lit)]

    def lit_command(self) -> list[str]:
        return self.lit_invocation() + [
            str(self.config.test_path),
            f"-j{self.config.workers}",
            "--no-progress-bar",
        ]

    def note_pinning(self) -> None:
        """Record the pinning that will apply to measured runs (metadata only).

        Called once after the environment context is entered, so subclasses can
        report a mode that depends on what setup() decided.
        """

    def _spawn_preexec(self) -> Callable[[], None] | None:
        """Return a callable run in the child before exec, or None.

        Used to pin CPU affinity / priority natively so no wrapper process is
        in the timed path. None means no per-child setup.
        """
        return None

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """Kill a spawned child and reap it; used on the error/abort path."""
        proc.kill()
        proc.wait()

    def measure_run(self, cmd: list[str], stage: str) -> RunSample:
        """Run cmd to completion; return its wall time and peak RSS.

        Wall time covers the lit process from just after spawn to exit; peak
        RSS comes from os.wait4 (None where wait4 is unavailable).
        """
        preexec = self._spawn_preexec()
        have_wait4 = hasattr(os, "wait4")
        if not have_wait4:
            log.warning(
                "os.wait4 unavailable on this platform; reporting wall clock only."
            )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=preexec,
            )
        except OSError as err:
            raise BenchmarkError(stage, f"Command failed: {' '.join(cmd)}") from err
        start = time.perf_counter()
        if not have_wait4:
            try:
                proc.wait()
            except BaseException:
                self._terminate(proc)
                raise
            wall = time.perf_counter() - start
            return RunSample(wall, None)
        try:
            _, wait_status, usage = os.wait4(proc.pid, 0)
        except BaseException:
            self._terminate(proc)
            raise
        wall = time.perf_counter() - start
        # wait4 already reaped the child; record the status so Popen does not
        # poll the freed pid (e.g. from __del__).
        proc.returncode = _exit_code_from_wait_status(wait_status)
        # os.wait4 reports ru_maxrss in kilobytes on Linux (and most POSIX
        # systems) but in bytes on macOS.
        rss_unit_bytes = 1 if SYSTEM == "Darwin" else 1024
        return RunSample(wall, usage.ru_maxrss * rss_unit_bytes)
