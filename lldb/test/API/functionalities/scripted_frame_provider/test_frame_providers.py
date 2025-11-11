"""
Test frame providers for scripted frame provider functionality.

These providers demonstrate various merge strategies:
- Replace: Replace entire stack
- Prepend: Add frames before real stack
- Append: Add frames after real stack

It also shows the ability to mix a dictionary, a ScriptedFrame or an SBFrame
index to create stackframes
"""

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
            return index - 2  # Return real frame index
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
            return index  # Return real frame index
        elif index == len(self.input_frames):
            return {
                "idx": 1,
                "pc": 0x10,
            }
        return None


class CustomScriptedFrame(ScriptedFrame):
    """Custom scripted frame with full control over frame behavior."""

    def __init__(self, thread, idx, pc, function_name):
        # Initialize structured data args
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
            return len(self.input_frames) - 2  # Real frame index
        elif index == 4:
            return len(self.input_frames) - 1  # Real frame index
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
