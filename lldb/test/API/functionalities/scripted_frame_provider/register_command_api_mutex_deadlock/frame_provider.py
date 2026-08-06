"""
Frame provider that returns dict-based synthetic frames (never identity
forwarding), while touching self.input_frames from get_frame_at_index.

Returning a dict keeps this test isolated from the frame-aliasing bug:
dict-based frames always go through ScriptedFrameProvider's
create_frame_from_dict helper, which builds a brand new StackFrame and
never reuses (or wraps via BorrowedStackFrame) the parent list's frame
object. Only the API-mutex deadlock is reachable through this path.
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
