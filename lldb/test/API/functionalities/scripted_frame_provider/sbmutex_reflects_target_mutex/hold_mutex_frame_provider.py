"""
Frame provider whose get_frame_at_index locks the target's real API
mutex via SBMutex and holds it briefly, from inside the bypassed
scripted-extension callback. See TestHoldMutexNoDeadlock.py for why this
must not deadlock.
"""

import time

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider

HOLD_DURATION_SECONDS = 0.2


class HoldMutexFrameProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return (
            "Provider that holds the real API mutex via SBMutex from get_frame_at_index"
        )

    def get_frame_at_index(self, index):
        if index >= len(self.input_frames):
            return None

        if index == 0:
            mutex = self.target.GetAPIMutex()
            mutex.lock()
            time.sleep(HOLD_DURATION_SECONDS)
            mutex.unlock()

        frame = self.input_frames[index]
        if frame is None:
            return None
        return {"idx": index, "pc": frame.GetPC()}
