"""
Test frame providers for scripted frame provider functionality.

These providers demonstrates various merge strategies:
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

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_stackframes(self, real_frames):
        return [
            {
                "idx": 0,
                "pc": 0x1000,
            },
            1,
            {
                "idx": 2,
                "pc": 0x3000,
            },
        ]


class PrependFrameProvider(ScriptedFrameProvider):
    """Prepend synthetic frames before real stack."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_stackframes(self, real_frames):
        l = [
            {
                "pc": 0x9000,
            },
            {
                "pc": 0xA000,
            },
        ]
        l.extend(list(range(0, len(real_frames))))
        return l


class AppendFrameProvider(ScriptedFrameProvider):
    """Append synthetic frames after real stack."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_stackframes(self, real_frames):
        l = list(range(0, len(real_frames)))
        l.append(
            {
                "idx": 1,
                "pc": 0x10,
            }
        )
        return l


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

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_stackframes(self, real_frames):
        """Return list of ScriptedFrame objects."""
        return [
            CustomScriptedFrame(self.thread, 0, 0x5000, "custom_scripted_frame_0"),
            {"pc": 0x6000},
            CustomScriptedFrame(self.thread, 2, 0x7000, "custom_scripted_frame_2"),
            len(real_frames) - 2,
            len(real_frames) - 1,
        ]
