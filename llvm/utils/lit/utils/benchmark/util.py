"""Shared low-level helpers with no benchmark-domain knowledge.

Leaf module: imports nothing from the package, so it can never participate
in an import cycle.
"""

from __future__ import annotations

import logging
import os
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from platform import system
from typing import TypedDict

SYSTEM = system()

log = logging.getLogger("benchmark")


class _LevelFormatter(logging.Formatter):
    """Plain message for INFO (status banners); 'LEVEL:' prefix above that."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_LevelFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


class BenchmarkError(Exception):
    """A benchmark stage failed; the cause travels via `raise ... from`."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"Stage '{stage}' failed: {message}")


def run_sys(
    cmd: list[str],
    stage: str,
    check: bool = True,
    capture: bool = False,
    text: bool | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=text if text is not None else capture,
            **kwargs,
        )
    except (subprocess.CalledProcessError, OSError) as err:
        err_msg = getattr(err, "stderr", None) or str(err)
        raise BenchmarkError(
            stage, f"Command failed: {' '.join(cmd)}\nDetails: {err_msg.strip()}"
        ) from err


def sudo_write(path: Path, value: str, stage: str) -> None:
    run_sys(
        ["sudo", "tee", str(path)],
        stage,
        input=value,
        text=True,
        stdout=subprocess.DEVNULL,
    )


def _exit_code_from_wait_status(wait_status: int) -> int:
    # FIXME: replace with os.waitstatus_to_exitcode() once lit's minimum
    # Python version reaches 3.9+
    # https://docs.python.org/3/library/os.html#os.waitstatus_to_exitcode
    if os.WIFEXITED(wait_status):
        return os.WEXITSTATUS(wait_status)
    if os.WIFSIGNALED(wait_status):
        return -os.WTERMSIG(wait_status)
    return -1


def format_bytes(num_bytes: float) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.1f} MiB"


def _fmt_duration(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


class Stats(TypedDict):
    """Reduction of a sample list; stddev is None for a single sample."""

    n: int
    mean: float
    median: float
    min: float
    max: float
    stddev: float | None


def summarize_samples(samples: Sequence[float]) -> Stats:
    """Reduce a list of measurements to n / mean / median / min / max / stddev."""
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stddev": statistics.stdev(samples) if len(samples) > 1 else None,
    }
