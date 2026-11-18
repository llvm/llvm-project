"""macOS platform: keep the host awake via caffeinate."""

from __future__ import annotations

import atexit
import subprocess

from ..config import BenchmarkConfig
from ..util import BenchmarkError, log
from .base import Platform


class MacPlatform(Platform):
    """Keep a macOS host awake. Darwin has no public affinity API, so runs
    are not CPU-pinned; expect more scheduling noise than Linux/Windows."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.proc: subprocess.Popen | None = None

    def setup(self) -> None:
        try:
            self.proc = subprocess.Popen(["caffeinate", "-dimsu"])
        except OSError as err:
            raise BenchmarkError("caffeinate", "could not start caffeinate") from err
        # Reap even if the benchmark process is torn down without restore() running.
        atexit.register(self._stop_caffeinate)
        self.applied.append("caffeinate")
        log.info("=== caffeinate started (no CPU pinning available on macOS) ===")

    def restore(self) -> None:
        self._stop_caffeinate()
        atexit.unregister(self._stop_caffeinate)

    def _stop_caffeinate(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.info("=== caffeinate stopped ===")
