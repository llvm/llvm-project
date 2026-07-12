"""
Frame provider that forwards every frame under its own index while also
touching input_frames.

Returning `index` (identity forwarding) means the provider reuses the
same StackFrame object that is cached in the parent (input) frame list
instead of wrapping it in a BorrowedStackFrame. Frame construction
unconditionally re-tags that frame as belonging to this (child) list,
corrupting the parent list's cached frame. When that corrupted frame is
later resolved back to a frame list, it resolves to this list -- which,
if the resolving thread is the one already fetching frames on this list,
self-deadlocks trying to take a reader lock on the writer lock it
already holds.
"""

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class IdentityProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "Provider that forwards each frame under its own index"

    def get_frame_at_index(self, index):
        if index < len(self.input_frames):
            # __getitem__ calls SBFrame.IsValid() internally, which is what
            # exercises GetStoppedExecutionContext.
            self.input_frames[index]
            return index
        return None
