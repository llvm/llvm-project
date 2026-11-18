"""Command-line interface: argument parsing, config building, and main entry point."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import BenchmarkConfig
from .platforms import make_platform
from .runner import BenchmarkRunner
from .util import BenchmarkError, _configure_logging, log


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return number


def _cpu_list(value: str) -> tuple[int, ...]:
    try:
        cpus = tuple(
            dict.fromkeys(int(part) for part in value.split(",") if part.strip())
        )
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid CPU list: {value!r}")
    if not cpus or any(cpu < 0 for cpu in cpus):
        raise argparse.ArgumentTypeError(f"invalid CPU list: {value!r}")
    return cpus


EPILOG = """\
example:
  python benchmark.py --repo-root ~/llvm-project --lit build/bin/llvm-lit \\
      --test-path build/test/Other --workers 4

notes:
  Wall clock and peak RSS are measured together, run for run. Linux setup needs
  sudo and an Intel CPU; Windows setup needs Administrator. --skip-env-setup
  skips machine tuning but keeps CPU pinning. Results go to
  <repo-root>/results/<label>-<timestamp>/.
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark wall-clock time and peak RSS of llvm-lit.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="LLVM checkout root; results are written under <repo-root>/results",
    )
    parser.add_argument(
        "--lit",
        required=True,
        help="path to llvm-lit (absolute, or relative to --repo-root)",
    )
    parser.add_argument(
        "--test-path",
        required=True,
        help="test file or directory to benchmark (absolute, or relative to --repo-root)",
    )
    parser.add_argument(
        "--workers", type=_positive_int, default=4, help="lit -j value (default: 4)"
    )
    parser.add_argument(
        "--label", default="run", help="results directory prefix (default: run)"
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=5,
        help="warmup runs discarded before measuring (default: 5)",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=10,
        help="measured runs (default: 10)",
    )
    parser.add_argument(
        "--benchmark-cpus",
        type=_cpu_list,
        default="2,4,6,8",
        help="CPUs to pin to on Linux/Windows (default: 2,4,6,8)",
    )
    parser.add_argument(
        "--disable-cset",
        action="store_true",
        help="use native CPU affinity instead of a cset shield on Linux",
    )
    parser.add_argument(
        "--skip-env-setup",
        action="store_true",
        help="skip machine tuning (no sudo/Administrator); CPU pinning still applies",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    repo_root = Path(args.repo_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return BenchmarkConfig(
        repo_root=repo_root,
        lit=repo_root / args.lit,
        test_path=repo_root / args.test_path,
        workers=args.workers,
        warmup=args.warmup,
        runs=args.runs,
        out_dir=repo_root / "results" / f"{args.label}-{stamp}",
        benchmark_cpus=args.benchmark_cpus,
        disable_cset=args.disable_cset,
        skip_env=args.skip_env_setup,
    )


def _validate_config(config: BenchmarkConfig) -> list[str]:
    """Return human-readable problems; empty means the config is runnable."""
    problems = []
    if not config.lit.exists():
        problems.append(f"lit not found at {config.lit}")
    if not config.test_path.exists():
        problems.append(f"test path not found at {config.test_path}")
    return problems


def main() -> None:
    _configure_logging()
    args = _parse_args()
    config = _build_config(args)
    problems = _validate_config(config)
    if problems:
        sys.exit("ERROR: " + "; ".join(problems))
    config.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        BenchmarkRunner(config, make_platform(config)).run()
    except KeyboardInterrupt:
        log.warning("Interrupted; environment was restored by the platform context.")
        raise SystemExit(130)
    except BenchmarkError as err:
        log.error(str(err), exc_info=True)
        raise SystemExit(1)
    log.info("=== Done ===")
