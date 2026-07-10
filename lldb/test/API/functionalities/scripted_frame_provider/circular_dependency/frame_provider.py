"""
Frame provider that reproduces the circular dependency issue.

This provider accesses input_frames and calls methods on them,
which before the fix would cause a circular dependency.
"""

import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class CustomScriptedFrame(ScriptedFrame):
    """Custom scripted frame with full control over frame behavior."""

    def __init__(self, thread, idx, pc, function_name):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.idx = idx
        self.pc = pc
        self.function_name = function_name

    def get_id(self):
        """Return the frame index."""
        return self.idx

    def get_pc(self):
        """Return the program counter."""
        return self.pc

    def get_function_name(self):
        """Return the function name."""
        return self.function_name

    def is_artificial(self):
        """Mark as artificial frame."""
        return False

    def is_hidden(self):
        """Not hidden."""
        return False

    def get_register_context(self):
        return None


class ScriptedFrameObjectProvider(ScriptedFrameProvider):
    """
    Provider that returns ScriptedFrame objects and accesses input_frames.

    This provider demonstrates the circular dependency bug fix:
    1. During get_frame_at_index(), we access input_frames[idx]
    2. We call frame.GetFunctionName() and frame.GetPC() on input frames
    3. Before the fix: These calls would trigger ExecutionContextRef::GetFrameSP()
       which would call Thread::GetStackFrameList() -> circular dependency!
    4. After the fix: ExecutionContextRef uses the remembered frame list -> no circular dependency
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        self.replacement_count = 0
        if self.target.process:
            baz_symbol_ctx = self.target.FindFunctions("baz")
            self.baz_symbol_ctx = None
            if len(baz_symbol_ctx) == 1:
                self.baz_symbol_ctx = baz_symbol_ctx[0]

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider that replaces 'bar' function with 'baz'"

    def get_frame_at_index(self, idx):
        """
        Replace frames named 'bar' with custom frames named 'baz'.

        This accesses input_frames and calls methods on them, which would
        trigger the circular dependency bug before the fix.
        """
        if idx < len(self.input_frames):
            # This access and method calls would cause circular dependency before fix!
            frame = self.input_frames[idx]

            # Calling GetFunctionName() triggers ExecutionContextRef resolution.
            function_name = frame.GetFunctionName()

            if function_name == "bar" and self.baz_symbol_ctx:
                # Replace "bar" with "baz".
                baz_func = self.baz_symbol_ctx.GetFunction()
                new_function_name = baz_func.GetName()
                pc = baz_func.GetStartAddress().GetLoadAddress(self.target)
                custom_frame = CustomScriptedFrame(
                    self.thread, idx, pc, new_function_name
                )
                self.replacement_count += 1
                return custom_frame

            # Pass through other frames by returning their index.
            return idx

        return None


class HandleCommandInInitProvider(ScriptedFrameProvider):
    """
    Provider that calls HandleCommand during __init__.

    This reproduces the circular dependency by running a command that
    accesses the thread's stack frames while the provider is being
    initialized (i.e. while the SyntheticStackFrameList is being built).
    Before the fix, HandleCommand("bt") would call Thread::GetStackFrameList()
    which would try to create the SyntheticStackFrameList again -> deadlock.
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        # Running "bt" during init triggers frame access on the thread,
        # which before the fix would cause a circular dependency.
        result = lldb.SBCommandReturnObject()
        self.thread.GetProcess().GetTarget().GetDebugger().GetCommandInterpreter().HandleCommand(
            "bt", result
        )
        self.init_succeeded = result.Succeeded()

    @staticmethod
    def get_description():
        return "Provider that calls HandleCommand('bt') in __init__"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            return idx
        return None


class EvaluateExpressionInGetFrameProvider(ScriptedFrameProvider):
    """
    Provider that calls EvaluateExpression in get_frame_at_index.

    This reproduces the circular dependency by evaluating an expression
    that accesses the thread's stack frames while the provider is fetching
    frames. Before the fix, EvaluateExpression would call
    Thread::GetStackFrameList() which would re-enter the
    SyntheticStackFrameList -> circular dependency.
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Provider that calls EvaluateExpression in get_frame_at_index"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            # Evaluating an expression that calls a function triggers frame
            # access on the thread, which before the fix would cause a
            # circular dependency.
            frame.EvaluateExpression("baz()")
            return idx
        return None
