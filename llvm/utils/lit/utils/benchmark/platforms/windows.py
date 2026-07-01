"""Windows platform: power/timer tuning and CPU affinity via win32 API."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time

from ..config import BenchmarkConfig, RunSample
from ..util import BenchmarkError, log, run_sys
from .base import Platform


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
