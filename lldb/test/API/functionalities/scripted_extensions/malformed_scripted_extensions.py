"""
Intentionally malformed scripted extensions used by
TestScriptedExtensionsDiagnostics.

Each class either omits a required abstract method or raises a Python
exception from one of its affordance methods. The corresponding test asserts
that LLDB surfaces those errors to the user instead of silently swallowing
them.
"""

# ---------------------------------------------------------------------------
# Scripted Process
# ---------------------------------------------------------------------------


class MissingMethodsScriptedProcess:
    """Missing required abstract method `read_memory_at_address`."""

    def __init__(self, exe_ctx, args):
        self.exe_ctx = exe_ctx
        self.args = args

    def get_scripted_thread_plugin(self):
        return None

    def is_alive(self):
        return True


class ExceptionScriptedProcess:
    """All abstract methods present, but `launch` raises."""

    def __init__(self, exe_ctx, args):
        self.exe_ctx = exe_ctx
        self.args = args

    def get_scripted_thread_plugin(self):
        return None

    def is_alive(self):
        return True

    def read_memory_at_address(self, addr, size, error):
        return None

    def launch(self):
        raise RuntimeError("intentional exception from launch()")


# ---------------------------------------------------------------------------
# Scripted Thread
# ---------------------------------------------------------------------------


class ExceptionScriptedThread:
    def __init__(self, process, args):
        self.process = process
        self.args = args

    def get_thread_id(self):
        raise ValueError("intentional exception from get_thread_id()")

    def get_register_context(self):
        return ""

    def get_name(self):
        return "ExceptionScriptedThread"

    def get_state(self):
        return 0


# ---------------------------------------------------------------------------
# Scripted Platform
# ---------------------------------------------------------------------------


class MissingMethodsScriptedPlatform:
    """Missing required abstract method `list_processes`."""

    def __init__(self, exe_ctx, args):
        self.exe_ctx = exe_ctx
        self.args = args


class ExceptionScriptedPlatform:
    def __init__(self, exe_ctx, args):
        self.exe_ctx = exe_ctx
        self.args = args

    def list_processes(self):
        raise RuntimeError("intentional exception from list_processes()")

    def get_process_info(self, pid):
        return None

    def launch_process(self, launch_info):
        return None

    def kill_process(self, pid):
        return None


# ---------------------------------------------------------------------------
# Scripted Frame Provider
# ---------------------------------------------------------------------------


class ExceptionScriptedFrameProvider:
    def __init__(self, frames, args):
        self.frames = frames
        self.args = args

    def get_num_frames(self):
        raise RuntimeError("intentional exception from get_num_frames()")

    def get_frame_at_index(self, idx):
        return None


# ---------------------------------------------------------------------------
# Scripted Thread Plan
# ---------------------------------------------------------------------------


class ExceptionScriptedThreadPlan:
    def __init__(self, thread_plan, args):
        self.thread_plan = thread_plan
        self.args = args

    def explains_stop(self, event):
        raise RuntimeError("intentional exception from explains_stop()")

    def should_stop(self, event):
        return True

    def is_stale(self):
        return False


# ---------------------------------------------------------------------------
# Scripted Breakpoint Resolver
# ---------------------------------------------------------------------------


class ExceptionScriptedBreakpointResolver:
    def __init__(self, bkpt, args):
        self.bkpt = bkpt
        self.args = args

    def __callback__(self, sym_ctx):
        raise RuntimeError("intentional exception from __callback__()")

    def get_short_help(self):
        return "Exception breakpoint resolver"


# ---------------------------------------------------------------------------
# Scripted Stop Hook
# ---------------------------------------------------------------------------


class ExceptionScriptedStopHook:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def handle_stop(self, exe_ctx, stream):
        raise RuntimeError("intentional exception from handle_stop()")


# ---------------------------------------------------------------------------
# Operating System
# ---------------------------------------------------------------------------


class MissingMethodsOperatingSystem:
    """Missing required abstract method `get_thread_info`."""

    def __init__(self, process):
        self.process = process


class ExceptionOperatingSystem:
    def __init__(self, process):
        self.process = process

    def get_thread_info(self):
        raise RuntimeError("intentional exception from get_thread_info()")

    def get_register_info(self):
        return {}

    def get_register_data(self, tid):
        return b""
