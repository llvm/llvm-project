"""
Test frame providers for scripted frame provider functionality.

These providers demonstrate various merge strategies:
- Replace: Replace entire stack
- Prepend: Add frames before real stack
- Append: Add frames after real stack

It also shows the ability to mix a dictionary, a ScriptedFrame or an SBFrame
index to create stackframes
"""

import os
import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class ReplaceFrameProvider(ScriptedFrameProvider):
    """Replace entire stack with custom frames."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        self.frames = [
            {
                "idx": 0,
                "pc": 0x1000,
            },
            0,
            {
                "idx": 2,
                "pc": 0x3000,
            },
        ]

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Replace entire stack with 3 custom frames"

    def get_frame_at_index(self, index):
        if index >= len(self.frames):
            return None
        return self.frames[index]


class PrependFrameProvider(ScriptedFrameProvider):
    """Prepend synthetic frames before real stack."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Prepend 2 synthetic frames before real stack"

    def get_frame_at_index(self, index):
        if index == 0:
            return {"pc": 0x9000}
        elif index == 1:
            return {"pc": 0xA000}
        elif index - 2 < len(self.input_frames):
            return index - 2  # Return real frame index.
        return None


class AppendFrameProvider(ScriptedFrameProvider):
    """Append synthetic frames after real stack."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Append 1 synthetic frame after real stack"

    def get_frame_at_index(self, index):
        if index < len(self.input_frames):
            return index  # Return real frame index.
        elif index == len(self.input_frames):
            return {
                "idx": 1,
                "pc": 0x10,
            }
        return None


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
        """No register context for this test."""
        return None


class ScriptedFrameObjectProvider(ScriptedFrameProvider):
    """Provider that returns ScriptedFrame objects instead of dictionaries."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider returning custom ScriptedFrame objects"

    def get_frame_at_index(self, index):
        """Return ScriptedFrame objects or dictionaries based on index."""
        if index == 0:
            return CustomScriptedFrame(
                self.thread, 0, 0x5000, "custom_scripted_frame_0"
            )
        elif index == 1:
            return {"pc": 0x6000}
        elif index == 2:
            return CustomScriptedFrame(
                self.thread, 2, 0x7000, "custom_scripted_frame_2"
            )
        elif index == 3:
            return len(self.input_frames) - 2  # Real frame index.
        elif index == 4:
            return len(self.input_frames) - 1  # Real frame index.
        return None


class ThreadFilterFrameProvider(ScriptedFrameProvider):
    """Provider that only applies to thread with ID 1."""

    @staticmethod
    def applies_to_thread(thread):
        """Only apply to thread with index ID 1."""
        return thread.GetIndexID() == 1

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider that only applies to thread ID 1"

    def get_frame_at_index(self, index):
        """Return a single synthetic frame."""
        if index == 0:
            return {"pc": 0xFFFF}
        return None


class CircularDependencyTestProvider(ScriptedFrameProvider):
    """
    Provider that tests the circular dependency fix.

    This provider accesses input_frames during __init__ and calls methods
    on those frames. Before the fix, this would cause a circular dependency:
    - Thread::GetStackFrameList() creates provider
    - Provider's __init__ accesses input_frames[0]
    - SBFrame::GetPC() tries to resolve ExecutionContextRef
    - ExecutionContextRef::GetFrameSP() calls Thread::GetStackFrameList()
    - Re-enters initialization -> circular dependency!

    With the fix, ExecutionContextRef remembers the frame list, so it doesn't
    re-enter Thread::GetStackFrameList().
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

        # This would cause circular dependency before the fix!
        # Accessing frames and calling methods on them during __init__
        self.original_frame_count = len(input_frames)
        self.original_pcs = []

        # Call GetPC() on each input frame - this triggers ExecutionContextRef resolution.
        for i in range(min(3, len(input_frames))):
            frame = input_frames[i]
            if frame.IsValid():
                pc = frame.GetPC()
                self.original_pcs.append(pc)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider that tests circular dependency fix by accessing frames in __init__"

    def get_frame_at_index(self, index):
        """Prepend a synthetic frame, then pass through original frames."""
        if index == 0:
            # Synthetic frame at index 0.
            return {"pc": 0xDEADBEEF}
        elif index - 1 < self.original_frame_count:
            # Pass through original frames at indices 1, 2, 3, ...
            return index - 1
        return None


class PythonSourceFrame(ScriptedFrame):
    """Scripted frame that points to Python source code."""

    def __init__(self, thread, idx, function_name, python_file, line_number):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.idx = idx
        self.function_name = function_name
        self.python_file = python_file
        self.line_number = line_number

    def get_id(self):
        """Return the frame index."""
        return self.idx

    def get_pc(self):
        """PC-less frame - return invalid address."""
        return lldb.LLDB_INVALID_ADDRESS

    def get_function_name(self):
        """Return the function name."""
        return self.function_name

    def get_symbol_context(self):
        """Return a symbol context with LineEntry pointing to Python source."""
        # Create a LineEntry pointing to the Python source file
        line_entry = lldb.SBLineEntry()
        line_entry.SetFileSpec(lldb.SBFileSpec(self.python_file, True))
        line_entry.SetLine(self.line_number)
        line_entry.SetColumn(0)

        # Create a symbol context with the line entry
        sym_ctx = lldb.SBSymbolContext()
        sym_ctx.SetLineEntry(line_entry)

        return sym_ctx

    def is_artificial(self):
        """Not artificial."""
        return False

    def is_hidden(self):
        """Not hidden."""
        return False

    def get_register_context(self):
        """No register context for PC-less frames."""
        return None


class PythonSourceFrameProvider(ScriptedFrameProvider):
    """
    Provider that demonstrates Python source display in scripted frames.

    This provider prepends frames pointing to Python source code, showing
    that PC-less frames can display Python source files with proper line
    numbers and module/compile unit information.
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

        # Find the python_helper.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.python_file = os.path.join(current_dir, "python_helper.py")

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider that prepends frames pointing to Python source"

    def get_frame_at_index(self, index):
        """Return Python source frames followed by original frames."""
        if index == 0:
            # Frame pointing to compute_fibonacci function (line 7)
            return PythonSourceFrame(
                self.thread, 0, "compute_fibonacci", self.python_file, 7
            )
        elif index == 1:
            # Frame pointing to process_data function (line 16)
            return PythonSourceFrame(
                self.thread, 1, "process_data", self.python_file, 16
            )
        elif index == 2:
            # Frame pointing to main function (line 27)
            return PythonSourceFrame(self.thread, 2, "main", self.python_file, 27)
        elif index - 3 < len(self.input_frames):
            # Pass through original frames
            return index - 3
        return None


class ValidPCNoModuleFrame(ScriptedFrame):
    """Scripted frame with a valid PC but no associated module."""

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
        """Not artificial."""
        return False

    def is_hidden(self):
        """Not hidden."""
        return False

    def get_register_context(self):
        """No register context."""
        return None


class ValidPCNoModuleFrameProvider(ScriptedFrameProvider):
    """
    Provider that demonstrates frames with valid PC but no module.

    This tests that backtrace output handles frames that have a valid
    program counter but cannot be resolved to any loaded module.
    """

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Provider that prepends frames with valid PC but no module"

    def get_frame_at_index(self, index):
        """Return frames with valid PCs but no module information."""
        if index == 0:
            # Frame with valid PC (0x1234000) but no module
            return ValidPCNoModuleFrame(self.thread, 0, 0x1234000, "unknown_function_1")
        elif index == 1:
            # Another frame with valid PC (0x5678000) but no module
            return ValidPCNoModuleFrame(self.thread, 1, 0x5678000, "unknown_function_2")
        elif index - 2 < len(self.input_frames):
            # Pass through original frames
            return index - 2
        return None


class AddFooFrameProvider(ScriptedFrameProvider):
    """Add a single 'foo' frame at the beginning."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Add 'foo' frame at beginning"

    @staticmethod
    def get_priority():
        """Return priority 10 (runs first in chain)."""
        return 10

    def get_frame_at_index(self, index):
        if index == 0:
            # Return synthetic "foo" frame
            return CustomScriptedFrame(self.thread, 0, 0xF00, "foo")
        elif index - 1 < len(self.input_frames):
            # Pass through input frames (shifted by 1)
            return index - 1
        return None


class AddBarFrameProvider(ScriptedFrameProvider):
    """Add a single 'bar' frame at the beginning."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Add 'bar' frame at beginning"

    @staticmethod
    def get_priority():
        """Return priority 20 (runs second in chain)."""
        return 20

    def get_frame_at_index(self, index):
        if index == 0:
            # Return synthetic "bar" frame
            return CustomScriptedFrame(self.thread, 0, 0xBAA, "bar")
        elif index - 1 < len(self.input_frames):
            # Pass through input frames (shifted by 1)
            return index - 1
        return None


class AddBazFrameProvider(ScriptedFrameProvider):
    """Add a single 'baz' frame at the beginning."""

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        """Return a description of this provider."""
        return "Add 'baz' frame at beginning"

    @staticmethod
    def get_priority():
        """Return priority 30 (runs last in chain)."""
        return 30

    def get_frame_at_index(self, index):
        if index == 0:
            # Return synthetic "baz" frame
            return CustomScriptedFrame(self.thread, 0, 0xBAC, "baz")
        elif index - 1 < len(self.input_frames):
            # Pass through input frames (shifted by 1)
            return index - 1
        return None
