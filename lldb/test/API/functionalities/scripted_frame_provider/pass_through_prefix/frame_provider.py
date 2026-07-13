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

    def __init__(self, thread, idx, function_name, prefix):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.idx = idx
        self.function_name = prefix + function_name

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
            return PrefixedFrame(self.thread, idx, function_name, self.PREFIX)
        return None


class UpperCasePassThroughProvider(ScriptedFrameProvider):
    """
    Provider that passes through every frame from its parent StackFrameList
    but upper-cases each function name.

    When chained after PrefixPassThroughProvider, the result should be
    e.g. 'MY_CUSTOM_BAZ'.
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Provider that upper-cases all function names"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            function_name = frame.GetFunctionName()
            return PrefixedFrame(self.thread, idx, function_name.upper(), "")
        return None


class BtProviderStarProvider(ScriptedFrameProvider):
    """
    Provider that runs 'bt --provider *' from within get_frame_at_index
    to verify that re-entrant provider queries don't deadlock or crash.

    On the first call to get_frame_at_index, it runs the command and stores
    the output. It then passes through all frames with a 'reentrant_' prefix.
    """

    PREFIX = "reentrant_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        self.bt_provider_star_output = None

    @staticmethod
    def get_description():
        return "Provider that runs 'bt --provider *' during frame construction"

    def get_frame_at_index(self, idx):
        if idx >= len(self.input_frames):
            return None

        # On the first frame request, run 'bt --provider *' re-entrantly.
        if self.bt_provider_star_output is None:
            debugger = self.target.GetDebugger()
            ci = debugger.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            ci.HandleCommand("bt --provider '*'", result)
            self.bt_provider_star_output = result.GetOutput() or ""

        frame = self.input_frames[idx]
        function_name = frame.GetFunctionName()
        return PrefixedFrame(self.thread, idx, function_name, self.PREFIX)


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

        # For frames after the first, peek at the younger (already-provided)
        # frame in input_frames. If it already has our prefix, we were handed
        # our own output list instead of the parent list.
        if idx > 0:
            younger = self.input_frames[idx - 1]
            if younger.GetFunctionName().startswith(self.PREFIX):
                return PrefixedFrame(
                    self.thread, idx, function_name, "danger_will_robinson_"
                )

        return PrefixedFrame(self.thread, idx, function_name, self.PREFIX)
