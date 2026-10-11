"""
Frame provider that returns dict-based synthetic frames while touching
self.input_frames from get_frame_at_index, to exercise the API-mutex
deadlock.
"""

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class DictFrameProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "Provider that returns dict-based synthetic frames"

    def get_frame_at_index(self, index):
        if index >= len(self.input_frames):
            return None
        # __getitem__ calls SBFrame.IsValid() internally, which is what
        # exercises GetStoppedExecutionContext.
        frame = self.input_frames[index]
        if frame is None:
            return None
        return {"idx": index, "pc": frame.GetPC()}
