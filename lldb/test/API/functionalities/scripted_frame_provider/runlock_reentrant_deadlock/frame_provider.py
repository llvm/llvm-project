"""
Frame provider whose get_frame_at_index calls SBFrame::IsValid on input frames.

Used to trigger the deadlock scenario described in
TestRunLockReentrantDeadlock.py: when a client thread accesses frames
through this provider, get_frame_at_index re-enters the SB API and tries
to re-acquire the ProcessRunLock read lock, which (without the recursion
fix in ProcessRunLocker) deadlocks against the override PST's pending
writer.
"""

import lldb
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class SBAPIAccessInGetFrameProvider(ScriptedFrameProvider):
    """Provider that calls SBFrame.IsValid from get_frame_at_index."""

    @staticmethod
    def get_description():
        return "Provider that accesses SB API in get_frame_at_index"

    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames.GetFrameAtIndex(idx)
            # This call triggers GetStoppedExecutionContext ->
            # ProcessRunLock::ReadTryLock. If the current thread already
            # holds the read lock (from the outer SB API entry point),
            # and a writer is pending, this re-entrant read lock blocks.
            frame.IsValid()
            return idx
        return None
