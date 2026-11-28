"""
Frame provider that forwards every input frame under its own index and counts
the frames it is asked for.
"""

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class PassThroughProvider(ScriptedFrameProvider):
    # The number of get_frame_at_index calls, over every instance.
    call_count = 0

    @staticmethod
    def get_description():
        return "Provider that forwards each frame under its own index"

    def get_frame_at_index(self, index):
        PassThroughProvider.call_count += 1
        if index < len(self.input_frames):
            return index
        return None
