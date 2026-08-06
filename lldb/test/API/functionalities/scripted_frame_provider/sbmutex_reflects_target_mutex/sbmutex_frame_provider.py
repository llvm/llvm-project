"""
Frame provider whose get_frame_at_index checks whether the target's real
API mutex is currently held by a different thread.

Used by TestSBMutexReflectsTargetMutex.py to confirm that SBMutex
(SBTarget::GetAPIMutex()) aliases the real, shared target mutex rather
than resolving to the no-op the bypass policy makes it for internal
callers.

The check must NOT run directly on this thread while still inside
get_frame_at_index: ScriptedPythonInterface::Dispatch pushes the
can_bypass_target_api_mutex policy for its entire duration, which spans
this whole Python call. TargetAPIMutex re-checks that policy on every
lock()/try_lock() call, so try_lock() from here would always resolve to
a genuine no-op -- no synchronization primitive touched at all, so it
can never observe contention from any other thread. That would make
"another thread held it" unreachable and the check meaningless,
regardless of what any other thread is actually doing to the real mutex.

Instead, the check runs on a plain background thread, spawned fresh from
here with no scripted-extension call on its stack. That thread never had
the bypass policy pushed, so TargetAPIMutex on it resolves to the
real, shared mutex from the start -- exactly the mutex a concurrently
running `bt` command (via CommandObjectParsed's eCommandTryTargetAPILock)
may be holding at that moment.

Only try_lock() is used, and it is never held beyond the immediate
check: an earlier version of this provider held the mutex for a short
duration to try to widen the race window, but that meant a genuinely
blocking acquisition from whichever thread invoked this callback -- not
every internal caller (e.g. the private state thread) already holds the
real mutex by the time it gets here, so that held the mutex, which
caused a real deadlock in practice. try_lock() never blocks, so this
cannot deadlock regardless of the outcome.
"""

import threading

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider


class ContentionCheckFrameProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "Provider that checks SBMutex contention from a background thread"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        self.artifact_path = None
        if self.args is not None:
            value = self.args.GetValueForKey("artifact_path")
            if value.IsValid():
                self.artifact_path = value.GetStringValue(4096)

    def _check_contention(self, mutex):
        # Runs on a fresh thread with no scripted-extension call (and so no
        # can_bypass_target_api_mutex) on its stack -- see module docstring.
        if mutex.try_lock():
            # Uncontended: nobody else holds the real mutex right now.
            # Undo the lock we just took.
            mutex.unlock()
            outcome = "no other thread held the real target API mutex"
        else:
            outcome = "another thread held the real target API mutex"
        with open(self.artifact_path, "a") as f:
            f.write(outcome + "\n")

    def get_frame_at_index(self, index):
        if index >= len(self.input_frames):
            return None

        if index == 0 and self.artifact_path:
            # Obtaining the mutex handle itself doesn't lock anything --
            # it's safe to do from inside the bypassed callback. Only the
            # actual try_lock() call, on the background thread, needs to
            # happen outside the bypass.
            mutex = self.target.GetAPIMutex()
            checker = threading.Thread(target=self._check_contention, args=(mutex,))
            checker.start()
            checker.join()

        frame = self.input_frames[index]
        if frame is None:
            return None
        return {"idx": index, "pc": frame.GetPC()}
