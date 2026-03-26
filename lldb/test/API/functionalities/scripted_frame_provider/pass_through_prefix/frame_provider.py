"""
Frame provider that passes through all frames but prefixes their function names.

This exercises the provider's ability to consult its parent StackFrameList
when constructing each frame, verifying that the push/pop mechanism correctly
routes frame lookups to the parent list.
"""

import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class PrefixedFrame(ScriptedFrame):
    """A frame that wraps a real frame but prefixes the function name."""

    def __init__(self, thread, idx, pc, function_name, prefix):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.idx = idx
        self.pc = pc
        self.function_name = prefix + function_name

    def get_id(self):
        return self.idx

    def get_pc(self):
        return self.pc

    def get_function_name(self):
        return self.function_name

    def is_artificial(self):
        return False

    def is_hidden(self):
        return False

    def get_register_context(self):
        return None


class PrefixPassThroughProvider(ScriptedFrameProvider):
    """
    Provider that passes through every frame from its parent StackFrameList
    but adds a prefix to each function name.

    This verifies that the provider can freely access its input_frames
    (the parent list) without hitting circular dependencies or deadlocks.
    """

    PREFIX = "my_custom_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Provider that prefixes all function names with 'my_custom_'"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            function_name = frame.GetFunctionName()
            pc = frame.GetPC()
            return PrefixedFrame(self.thread, idx, pc, function_name, self.PREFIX)
        return None


class ValidatingPrefixProvider(ScriptedFrameProvider):
    """
    Provider that prefixes function names AND validates it receives the
    parent StackFrameList (not its own output).

    When constructing frame N (where N > 0), peeks at input_frames[N-1].
    If that younger frame's name already carries the prefix, the provider
    was incorrectly given its own output list — it flags this by prepending
    'danger_will_robinson_' to the function name.
    """

    PREFIX = "my_custom_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Validating provider that detects wrong input list"

    def get_frame_at_index(self, idx):
        if idx >= len(self.input_frames):
            return None

        frame = self.input_frames[idx]
        function_name = frame.GetFunctionName()
        pc = frame.GetPC()

        # For frames after the first, peek at the younger (already-provided)
        # frame in input_frames. If it already has our prefix, we were handed
        # our own output list instead of the parent list.
        if idx > 0:
            younger = self.input_frames[idx - 1]
            if younger.GetFunctionName().startswith(self.PREFIX):
                return PrefixedFrame(
                    self.thread, idx, pc, function_name, "danger_will_robinson_"
                )

        return PrefixedFrame(self.thread, idx, pc, function_name, self.PREFIX)
