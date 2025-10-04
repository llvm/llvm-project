"""
Test frame providers for scripted frame provider functionality.

These providers exercise all merge strategies:
- Replace: Replace entire stack
- Prepend: Add frames before real stack
- Append: Add frames after real stack
- ReplaceByIndex: Replace specific frames by index
"""

import lldb
from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class ReplaceFrameProvider(ScriptedFrameProvider):
    """Replace entire stack with custom frames."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_merge_strategy(self):
        return lldb.eScriptedFrameProviderMergeStrategyReplace

    def get_stackframes(self):
        return [
            {
                "idx": 0,
                "pc": 0x1000,
            },
            {
                "idx": 1,
                "pc": 0x2000,
            },
            {
                "idx": 2,
                "pc": 0x3000,
            },
        ]


class PrependFrameProvider(ScriptedFrameProvider):
    """Prepend synthetic frames before real stack."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_merge_strategy(self):
        return lldb.eScriptedFrameProviderMergeStrategyPrepend

    def get_stackframes(self):
        # Get real frame 0 PC
        real_frame_0 = self.thread.GetFrameAtIndex(0)
        real_pc = (
            real_frame_0.GetPC() if real_frame_0 and real_frame_0.IsValid() else 0x1000
        )

        return [
            {
                "idx": 0,
                "pc": real_pc,
            },
            {
                "idx": 1,
                "pc": real_pc - 0x10,
            },
        ]


class AppendFrameProvider(ScriptedFrameProvider):
    """Append synthetic frames after real stack."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_merge_strategy(self):
        return lldb.eScriptedFrameProviderMergeStrategyAppend

    def get_stackframes(self):
        # Count real frames
        num_real_frames = self.thread.GetNumFrames()

        return [
            {
                "idx": num_real_frames,
                "pc": 0x9000,
            },
            {
                "idx": num_real_frames + 1,
                "pc": 0xA000,
            },
        ]


class ReplaceByIndexFrameProvider(ScriptedFrameProvider):
    """Replace only frames 0 and 2, keep frame 1 real."""

    def __init__(self, thread, args):
        super().__init__(thread, args)

    def get_merge_strategy(self):
        return lldb.eScriptedFrameProviderMergeStrategyReplaceByIndex

    def get_stackframes(self):
        frames = []

        # Replace frame 0
        real_frame_0 = self.thread.GetFrameAtIndex(0)
        if real_frame_0 and real_frame_0.IsValid():
            frames.append(
                {
                    "idx": 0,
                    "pc": real_frame_0.GetPC(),
                }
            )

        # Replace frame 2
        real_frame_2 = self.thread.GetFrameAtIndex(2)
        if real_frame_2 and real_frame_2.IsValid():
            frames.append(
                {
                    "idx": 2,
                    "pc": real_frame_2.GetPC(),
                }
            )

        return frames


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
        return True

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

    def get_merge_strategy(self):
        return lldb.eScriptedFrameProviderMergeStrategyReplace

    def get_stackframes(self):
        """Return list of ScriptedFrame objects."""
        return [
            CustomScriptedFrame(self.thread, 0, 0x5000, "custom_scripted_frame_0"),
            CustomScriptedFrame(self.thread, 1, 0x6000, "custom_scripted_frame_1"),
            CustomScriptedFrame(self.thread, 2, 0x7000, "custom_scripted_frame_2"),
        ]
