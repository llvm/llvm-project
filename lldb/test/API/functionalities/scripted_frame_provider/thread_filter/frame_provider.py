"""
Frame providers for multithreaded thread-filter testing.

EvenThreadProvider  - applies only to even-indexed threads, prepends 'even_thread_'.
OddThreadProvider   - applies only to odd-indexed threads, prepends 'odd_thread_'.
UpperCaseProvider   - applies to ALL threads, upper-cases function names.

When all three are registered (even, odd, uppercase -- in that order), chaining
produces:
  Even threads: EVEN_THREAD_THREAD_WORK
  Odd threads:  ODD_THREAD_THREAD_WORK
"""

import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class PrefixedFrame(ScriptedFrame):
    """A frame that wraps a real frame but transforms the function name."""

    def __init__(self, thread, idx, function_name):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)
        self.idx = idx
        self.function_name = function_name

    def get_id(self):
        return self.idx

    def get_pc(self):
        return 0

    def get_function_name(self):
        return self.function_name

    def is_artificial(self):
        return False

    def is_hidden(self):
        return False

    def get_register_context(self):
        return None


class EvenThreadProvider(ScriptedFrameProvider):
    """Applies only to even-indexed threads; prepends 'even_thread_' prefix."""

    PREFIX = "even_thread_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def applies_to_thread(thread):
        return thread.GetIndexID() % 2 == 0

    @staticmethod
    def get_description():
        return "Prefix for even threads"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            name = self.PREFIX + frame.GetFunctionName()
            return PrefixedFrame(self.thread, idx, name)
        return None


class OddThreadProvider(ScriptedFrameProvider):
    """Applies only to odd-indexed threads; prepends 'odd_thread_' prefix."""

    PREFIX = "odd_thread_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def applies_to_thread(thread):
        return thread.GetIndexID() % 2 != 0

    @staticmethod
    def get_description():
        return "Prefix for odd threads"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            name = self.PREFIX + frame.GetFunctionName()
            return PrefixedFrame(self.thread, idx, name)
        return None


class UpperCaseProvider(ScriptedFrameProvider):
    """Applies to ALL threads; upper-cases all function names."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Upper-case all function names"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            name = frame.GetFunctionName().upper()
            return PrefixedFrame(self.thread, idx, name)
        return None
