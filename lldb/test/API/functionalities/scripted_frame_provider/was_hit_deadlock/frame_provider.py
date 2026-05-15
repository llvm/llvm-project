"""
Scripted frame provider that accesses the SB API in __init__.

This provider checks SBThread validity during construction, which calls
GetStoppedExecutionContext and tries to acquire a recursive_mutex.
When combined with the scripted breakpoint's was_hit callback doing
EvaluateExpression on the private state thread, this causes a deadlock:

- Thread A (private state): holds the mutex in RunThreadPlan/WaitForProcessToStop
- Thread B (override state): loads this provider, calls SBThread.__bool__ ->
  GetStoppedExecutionContext -> tries to lock the same mutex -> DEADLOCK
"""

import lldb
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class SBAPIAccessInInitProvider(ScriptedFrameProvider):
    """Provider that accesses SBThread (triggering mutex acquisition) in __init__."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        # This is the key line that triggers the deadlock: checking if the
        # thread is valid calls SBThread::operator bool() ->
        # GetStoppedExecutionContext -> recursive_mutex::lock().
        # When the private state thread holds that mutex (during
        # RunThreadPlan/WaitForProcessToStop from a was_hit callback),
        # this blocks forever.
        if self.thread:
            self.thread_is_valid = bool(self.thread)
        else:
            self.thread_is_valid = False

    @staticmethod
    def get_description():
        return "Provider that accesses SB API in __init__ (deadlock reproducer)"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            return idx
        return None
