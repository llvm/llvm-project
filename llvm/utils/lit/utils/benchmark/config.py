"""Input parameters and per-run measurement-result data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSample:
    """One measured lit run: wall-clock seconds and peak RSS in bytes.

    peak_rss_bytes is None when the platform cannot report peak RSS for the
    run; the wall-clock time is always present.
    """

    wall_sec: float
    peak_rss_bytes: int | None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark parameters (built by _build_config, validated by _validate_config)."""

    repo_root: Path
    lit: Path
    test_path: Path
    workers: int
    warmup: int
    runs: int
    out_dir: Path
    benchmark_cpus: tuple[int, ...]
    disable_cset: bool
    skip_env: bool

    @property
    def cpu_list(self) -> str:
        return ",".join(str(cpu) for cpu in self.benchmark_cpus)

    @property
    def summary_json(self) -> Path:
        return self.out_dir / "summary.json"
