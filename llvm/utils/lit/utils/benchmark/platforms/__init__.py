"""Per-OS strategy registry and factory."""

from __future__ import annotations

from ..config import BenchmarkConfig
from ..util import SYSTEM, log
from .base import Platform
from .linux import LinuxPlatform
from .macos import MacPlatform
from .windows import WindowsPlatform

PLATFORMS: dict[str, type[Platform]] = {
    "Linux": LinuxPlatform,
    "Darwin": MacPlatform,
    "Windows": WindowsPlatform,
}


def make_platform(config: BenchmarkConfig, system_name: str = SYSTEM) -> Platform:
    """Build the per-OS strategy; parameterized on system_name for testing."""
    platform_cls = PLATFORMS.get(system_name)
    if platform_cls is None:
        log.warning(
            f"Unrecognized platform '{system_name}': no environment tuning or CPU "
            "pinning; results will be noisier."
        )
        platform_cls = Platform
    plat = platform_cls(config)
    if plat.PINS_CPUS and config.workers > len(config.benchmark_cpus):
        log.warning(
            f"--workers {config.workers} exceeds the {len(config.benchmark_cpus)} pinned "
            f"CPUs ({config.cpu_list}); lit workers will contend for cores."
        )
    return plat
