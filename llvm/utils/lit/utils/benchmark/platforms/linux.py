"""Linux platform: performance governor, turbo, SMT, ASLR, and CPU pinning."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..config import BenchmarkConfig
from ..util import BenchmarkError, log, run_sys, sudo_write
from .base import Platform


class LinuxPlatform(Platform):
    """Stabilize an Intel Linux host: performance governor, no turbo, no SMT,
    no ASLR, and CPU pinning via native sched_setaffinity (cset shield used
    for stronger isolation when available)."""

    PINS_CPUS = True

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.saved: dict[str, str] = {}
        self.use_cset = shutil.which("cset") is not None and not config.disable_cset
        # cset_active: benchmark process has joined the shield, so children inherit the
        # shielded cpuset and no preexec affinity is needed.
        self.cset_active = False
        # shield_created: a cset shield exists and must be reset on restore.
        self.shield_created = False
        if not self.use_cset:
            available = os.sched_getaffinity(0)  # type: ignore[attr-defined]
            missing = [c for c in config.benchmark_cpus if c not in available]
            if missing:
                log.warning(
                    f"CPUs {missing} are not in this process's affinity set "
                    f"{sorted(available)}; native pinning will not apply to them."
                )

    def setup(self) -> None:
        self._set_performance_governor()
        self._disable_turbo()
        self._disable_smt()
        self._disable_aslr()
        self._shield_cpus()

    def restore(self) -> None:
        self._unshield_cpus()
        for path, value in reversed(list(self.saved.items())):
            try:
                sudo_write(Path(path), value, f"Restore {path}")
                log.info(f"=== restored {path} => {value!r} ===")
            except BenchmarkError as err:
                log.error(f"Could not restore {path}: {err}")
        self.saved.clear()

    def note_pinning(self) -> None:
        if self.cset_active:
            return  # already recorded as cset-shield in _shield_cpus
        note = f"sched_setaffinity={self.config.cpu_list}"
        if os.geteuid() == 0:
            note += ",nice=-20"
        self.applied.append(note)

    def _spawn_preexec(self):
        # Inside an active cset shield the inherited cpuset already confines the
        # child, so no per-child affinity call is needed.
        if self.cset_active:
            return None
        cpus = set(self.config.benchmark_cpus)
        boost = os.geteuid() == 0

        def _child_setup() -> None:
            # Runs in the child after fork, before exec: pin affinity (so lit
            # and the workers it spawns inherit it) and, when privileged,
            # raise priority. Swallow errors; the run still proceeds unpinned
            # rather than aborting the spawn.
            try:
                os.sched_setaffinity(0, cpus)  # type: ignore[attr-defined]
            except OSError:
                pass
            if boost:
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, -20)
                except OSError:
                    pass

        return _child_setup

    def _sysfs_set(
        self, name: str, path: Path, value: str, optional: bool = False
    ) -> bool:
        """Save the current value of path and overwrite it; True if applied.

        No exists() pre-check: attempting the read and catching the error
        avoids the check-then-use gap (CWE-367).
        """
        try:
            original = path.read_text().strip()
        except FileNotFoundError:
            if optional:
                log.info(f"{path} not present; skipping {name}.")
                return False
            raise BenchmarkError(f"Set {name}", f"required path {path} does not exist")
        except OSError as err:
            raise BenchmarkError(f"Set {name}", f"could not read {path}") from err
        sudo_write(path, value, f"Set {name}")
        self.saved[str(path)] = original
        log.info(f"=== {name}: {original!r} => {value!r} ===")
        return True

    def _set_performance_governor(self) -> None:
        # Save/set every cpufreq policy individually so mixed per-core
        # governors restore faithfully; no cpupower dependency.
        cpufreq_dir = Path("/sys/devices/system/cpu/cpufreq")
        policies = sorted(cpufreq_dir.glob("policy*/scaling_governor"))
        if not policies:
            raise BenchmarkError(
                "Set governor", f"no cpufreq policies under {cpufreq_dir}"
            )
        for policy in policies:
            self._sysfs_set(f"governor:{policy.parent.name}", policy, "performance")
        self.applied.append(f"governor=performance ({len(policies)} policies)")

    def _disable_turbo(self) -> None:
        self._sysfs_set(
            "turbo", Path("/sys/devices/system/cpu/intel_pstate/no_turbo"), "1"
        )
        self.applied.append("turbo=off")

    def _disable_smt(self) -> None:
        # Optional: not every CPU exposes SMT control.
        if self._sysfs_set(
            "smt", Path("/sys/devices/system/cpu/smt/control"), "off", optional=True
        ):
            self.applied.append("smt=off")

    def _disable_aslr(self) -> None:
        self._sysfs_set("aslr", Path("/proc/sys/kernel/randomize_va_space"), "0")
        self.applied.append("aslr=off")

    def _shield_cpus(self) -> None:
        """Create a cset shield and move the benchmark process into it.

        The benchmark process joins the shield once so every child it later spawns
        inherits the shielded cpuset; pinning then costs nothing per run. If
        cset is unavailable or any step fails, fall back to native affinity
        (applied per child via _spawn_preexec).
        """
        if not self.use_cset:
            return
        res = run_sys(
            ["sudo", "cset", "shield", "-c", self.config.cpu_list, "-k", "on"],
            "CPU Shield",
            check=False,
            capture=True,
        )
        if res.returncode != 0:
            log.warning(
                f"cset shield failed; using native CPU affinity. Reason: {res.stderr.strip()}"
            )
            return
        self.shield_created = True
        if self._join_shield():
            self.cset_active = True
            self.applied.append(f"cset-shield={self.config.cpu_list}")
            log.info(
                f"=== cset shield on CPUs {self.config.cpu_list}; benchmark process joined ==="
            )
        else:
            # Could not join; drop the shield and fall back to native affinity.
            self._unshield_cpus()

    def _join_shield(self) -> bool:
        res = run_sys(
            [
                "sudo",
                "cset",
                "shield",
                "--shield",
                "--threads",
                "--pid",
                str(os.getpid()),
            ],
            "Join Shield",
            check=False,
            capture=True,
        )
        if res.returncode != 0:
            log.warning(
                "Could not move benchmark process into cset shield; using native CPU "
                f"affinity. Reason: {res.stderr.strip()}"
            )
            return False
        return True

    def _unshield_cpus(self) -> None:
        if not self.shield_created:
            return
        try:
            run_sys(["sudo", "cset", "shield", "--reset"], "Unshield CPU", check=False)
            log.info("=== cset shield removed ===")
        except BenchmarkError as err:
            log.error(f"Could not unshield CPUs: {err}")
        self.shield_created = False
        self.cset_active = False
