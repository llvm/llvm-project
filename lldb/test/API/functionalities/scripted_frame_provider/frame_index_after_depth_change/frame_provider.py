"""
Frame provider that forwards every input frame under its own index.

Identity forwarding still wraps each frame in a BorrowedStackFrame, which is
what this test needs: those wrappers end up cached in the thread's public frame
list, and that list becomes the predecessor the next unwinder list merges
against.
"""

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class IdentityProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "Provider that forwards each frame under its own index"

    def get_frame_at_index(self, index):
        if index < len(self.input_frames):
            return index
        return None
