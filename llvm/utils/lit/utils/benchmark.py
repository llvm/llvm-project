# ===------------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
"""
Platform assumptions:
  Linux - x86-64 Intel CPU, sudo; use --skip-env-setup if unsupported
  macOS - caffeinate available; no CPU-pinning API
  Windows - x86-64, env setup requires Administrator
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from platform import machine, release, system
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


class _WindowsNative:
    """Typed ctypes bindings for the win32 calls the benchmark needs.

    Keeps handle types, structures, and error decoding out of the platform
    policy class. Only constructed on Windows.
    """

    PROCESS_SET_INFORMATION = 0x0200
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    THREAD_SUSPEND_RESUME = 0x0002
    TH32CS_SNAPTHREAD = 0x00000004
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ThreadID", ctypes.c_uint32),
            ("th32OwnerProcessID", ctypes.c_uint32),
            ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long),
            ("dwFlags", ctypes.c_uint32),
        ]

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def __init__(self):
        # Windows-only ctypes attributes; this class is never constructed on
        # other platforms. The type: ignore comments matter only when type
        # checking on a non-Windows host.
        self.kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        self.shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        self.winmm = ctypes.WinDLL("winmm")  # type: ignore[attr-defined]
        self._declare_prototypes()

    def _declare_prototypes(self) -> None:
        k = self.kernel32
        # Handles are pointer-sized; ctypes' default c_int restype would
        # truncate them on 64-bit Windows.
        k.OpenProcess.restype = ctypes.c_void_p
        k.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k.OpenThread.restype = ctypes.c_void_p
        k.OpenThread.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        k.CloseHandle.argtypes = (ctypes.c_void_p,)
        k.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        k.ResumeThread.restype = ctypes.c_uint32
        k.ResumeThread.argtypes = (ctypes.c_void_p,)
        k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k.CreateToolhelp32Snapshot.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
        k.Thread32First.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(self._THREADENTRY32),
        )
        k.Thread32Next.argtypes = (ctypes.c_void_p, ctypes.POINTER(self._THREADENTRY32))
        k.K32GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(self._PROCESS_MEMORY_COUNTERS),
            ctypes.c_uint32,
        )

    def is_user_admin(self) -> bool:
        try:
            return bool(self.shell32.IsUserAnAdmin())
        except Exception:
            return False

    def begin_timer_period(self) -> None:
        self.winmm.timeBeginPeriod(1)

    def end_timer_period(self) -> None:
        self.winmm.timeEndPeriod(1)

    def open_process(self, pid: int, access: int) -> int | None:
        return self.kernel32.OpenProcess(access, False, pid)

    def set_affinity(self, handle: int, mask: int) -> bool:
        return bool(self.kernel32.SetProcessAffinityMask(handle, mask))

    def close_handle(self, handle: int) -> None:
        self.kernel32.CloseHandle(handle)

    def resume_primary_thread(self, pid: int) -> None:
        """Resume the single suspended thread of a CREATE_SUSPENDED process."""
        snapshot = self.kernel32.CreateToolhelp32Snapshot(self.TH32CS_SNAPTHREAD, 0)
        if snapshot is None or snapshot == self.INVALID_HANDLE_VALUE:
            raise ctypes.WinError()  # type: ignore[attr-defined]
        try:
            entry = self._THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            found = self.kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.th32OwnerProcessID == pid:
                    thread = self.kernel32.OpenThread(
                        self.THREAD_SUSPEND_RESUME, False, entry.th32ThreadID
                    )
                    if thread is None:
                        raise ctypes.WinError()  # type: ignore[attr-defined]
                    try:
                        if self.kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError()  # type: ignore[attr-defined]
                    finally:
                        self.kernel32.CloseHandle(thread)
                    return
                found = self.kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self.kernel32.CloseHandle(snapshot)
        raise OSError(f"no thread found for pid {pid}")

    def peak_working_set_bytes(self, handle: int) -> int | None:
        counters = self._PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = self.kernel32.K32GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            log.error(f"K32GetProcessMemoryInfo failed: {ctypes.WinError()}")  # type: ignore[attr-defined]
            return None
        return int(counters.PeakWorkingSetSize)


class WindowsPlatform(Platform):
    """Tune Windows power/timer state (needs Administrator) and pin the
    benchmark process via the win32 API."""

    PINS_CPUS = True
    RSS_SEMANTICS = (
        "peak working set of the lit driver process only "
        "(child worker processes not counted)"
    )

    BENCH_SCHEME = "SCHEME_MIN"
    SUB_PROCESSOR = "SUB_PROCESSOR"
    BOOST_SUBGROUP = "54533251-82be-4824-96c1-47b60b740d00"
    BOOST_SETTING = "be337238-0d82-4146-a960-4f3749d470c7"
    THROTTLE_SETTINGS = ("PROCTHROTTLEMIN", "PROCTHROTTLEMAX", "CPMINCORES")
    CREATE_SUSPENDED = 0x00000004

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.saved_power: dict[str, dict[str, str]] = {}
        self.saved_scheme: str | None = None
        self.timer_active = False
        self._native: _WindowsNative | None = None

    @property
    def native(self) -> _WindowsNative:
        if self._native is None:
            self._native = _WindowsNative()
        return self._native

    def setup(self) -> None:
        if not self.native.is_user_admin():
            raise BenchmarkError(
                "Windows Setup",
                "Administrator privileges required (or use --skip-env-setup)",
            )
        self._set_performance()
        self._disable_turbo()
        self._exclude_defender()

    def restore(self) -> None:
        self._restore_defender()
        self._restore_turbo()
        self._restore_performance()

    def note_pinning(self) -> None:
        self.applied.append(f"affinity-mask={self.config.cpu_list}")

    def _activate_scheme(self, scheme: str) -> None:
        run_sys(
            ["powercfg", "/setactive", scheme], "Activate Power Scheme", check=False
        )

    def _query_setting(self, sub: str, setting: str) -> dict[str, str]:
        res = run_sys(
            ["powercfg", "/query", self.BENCH_SCHEME, sub, setting],
            f"Query {setting}",
            capture=True,
        )
        out: dict[str, str] = {}
        for line in res.stdout.splitlines():
            if "Current AC Power Setting Index:" in line:
                out["ac"] = str(int(line.split(":")[-1].strip(), 16))
            elif "Current DC Power Setting Index:" in line:
                out["dc"] = str(int(line.split(":")[-1].strip(), 16))
        return out

    def _apply_setting(self, sub: str, setting: str, value: str) -> None:
        run_sys(
            ["powercfg", "/setacvalueindex", self.BENCH_SCHEME, sub, setting, value],
            f"Set AC {setting}",
        )
        run_sys(
            ["powercfg", "/setdcvalueindex", self.BENCH_SCHEME, sub, setting, value],
            f"Set DC {setting}",
        )

    def _restore_setting(self, key: str, sub: str, setting: str) -> None:
        saved = self.saved_power.get(key, {})
        try:
            if "ac" in saved:
                run_sys(
                    [
                        "powercfg",
                        "/setacvalueindex",
                        self.BENCH_SCHEME,
                        sub,
                        setting,
                        saved["ac"],
                    ],
                    "Restore AC",
                    check=False,
                )
            if "dc" in saved:
                run_sys(
                    [
                        "powercfg",
                        "/setdcvalueindex",
                        self.BENCH_SCHEME,
                        sub,
                        setting,
                        saved["dc"],
                    ],
                    "Restore DC",
                    check=False,
                )
        except BenchmarkError as err:
            log.error(f"Could not restore setting {setting}: {err}")

    def _set_performance(self) -> None:
        res = run_sys(
            ["powercfg", "/getactivescheme"], "Get Active Power Scheme", capture=True
        )
        parts = res.stdout.split()
        if len(parts) >= 4:
            self.saved_scheme = parts[3]
        for setting in self.THROTTLE_SETTINGS:
            self.saved_power[setting] = self._query_setting(self.SUB_PROCESSOR, setting)
            self._apply_setting(self.SUB_PROCESSOR, setting, "100")
        self._activate_scheme(self.BENCH_SCHEME)
        try:
            self.native.begin_timer_period()
            self.timer_active = True
        except Exception as err:
            raise BenchmarkError(
                "Timer Resolution", "could not set 1 ms timer resolution"
            ) from err
        self.applied.append("power-scheme=SCHEME_MIN,throttle=100%,timer=1ms")

    def _restore_performance(self) -> None:
        if self.timer_active:
            try:
                self.native.end_timer_period()
            except Exception as err:
                log.error(f"Could not restore timer resolution: {err}")
            self.timer_active = False
        for setting in self.THROTTLE_SETTINGS:
            self._restore_setting(setting, self.SUB_PROCESSOR, setting)
        if self.saved_scheme is not None:
            self._activate_scheme(self.saved_scheme)

    def _disable_turbo(self) -> None:
        self.saved_power["boost"] = self._query_setting(
            self.BOOST_SUBGROUP, self.BOOST_SETTING
        )
        self._apply_setting(self.BOOST_SUBGROUP, self.BOOST_SETTING, "0")
        self._activate_scheme("SCHEME_CURRENT")
        self.applied.append("boost=off")

    def _restore_turbo(self) -> None:
        self._restore_setting("boost", self.BOOST_SUBGROUP, self.BOOST_SETTING)
        self._activate_scheme("SCHEME_CURRENT")

    def _exclude_defender(self) -> None:
        run_sys(
            [
                "powershell",
                "-Command",
                f"Add-MpPreference -ExclusionPath '{self.config.repo_root}'",
            ],
            "Disable Defender",
        )
        self.applied.append("defender-exclusion")

    def _restore_defender(self) -> None:
        try:
            run_sys(
                [
                    "powershell",
                    "-Command",
                    f"Remove-MpPreference -ExclusionPath '{self.config.repo_root}'",
                ],
                "Restore Defender",
                check=False,
            )
        except BenchmarkError as err:
            log.error(f"Could not restore Defender settings: {err}")

    def lit_invocation(self) -> list[str]:
        # The build emits bin/llvm-lit.py without a shebang on Windows;
        # CreateProcess will not execute it by path, so run it via Python.
        return [sys.executable, str(self.config.lit)]

    def _affinity_mask(self) -> int:
        mask = 0
        for cpu in self.config.benchmark_cpus:
            if cpu >= 64:
                raise BenchmarkError(
                    "Affinity", f"CPU {cpu} exceeds the 64-bit affinity mask range"
                )
            mask |= 1 << cpu
        return mask

    def _spawn_suspended_pinned(
        self, cmd: list[str], stage: str, **popen_kwargs
    ) -> tuple[subprocess.Popen, int | None]:
        """Start cmd suspended at high priority and pin it; do not resume it.

        The suspended start closes the window where the process (and any
        children it spawns) would run unpinned before SetProcessAffinityMask
        lands. The caller resumes the primary thread (after starting its timer)
        and closes the returned process handle.
        """
        flags = subprocess.HIGH_PRIORITY_CLASS | self.CREATE_SUSPENDED  # type: ignore[attr-defined]
        try:
            proc = subprocess.Popen(cmd, creationflags=flags, **popen_kwargs)
        except OSError as err:
            raise BenchmarkError(stage, f"could not start: {' '.join(cmd)}") from err
        access = (
            _WindowsNative.PROCESS_SET_INFORMATION
            | _WindowsNative.PROCESS_QUERY_INFORMATION
            | _WindowsNative.PROCESS_VM_READ
        )
        handle = self.native.open_process(proc.pid, access)
        if handle is None:
            log.warning("OpenProcess failed; benchmark process runs unpinned.")
        elif not self.native.set_affinity(handle, self._affinity_mask()):
            log.warning(
                "SetProcessAffinityMask failed; benchmark process runs unpinned."
            )
        return proc, handle

    def measure_run(self, cmd: list[str], stage: str) -> RunSample:
        proc, handle = self._spawn_suspended_pinned(
            cmd, stage, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            # Time from resume so the suspended setup (spawn + pin) is excluded.
            start = time.perf_counter()
            try:
                self.native.resume_primary_thread(proc.pid)
            except OSError as err:
                self._terminate(proc)
                raise BenchmarkError(
                    stage, "could not resume the suspended benchmark process"
                ) from err
            try:
                proc.wait()
            except BaseException:
                self._terminate(proc)
                raise
            wall = time.perf_counter() - start
            # Peak counters remain queryable on an exited process while a
            # handle stays open.
            peak = (
                self.native.peak_working_set_bytes(handle)
                if handle is not None
                else None
            )
            return RunSample(wall, peak)
        finally:
            if handle is not None:
                self.native.close_handle(handle)


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


if __name__ == "__main__":
    main()
