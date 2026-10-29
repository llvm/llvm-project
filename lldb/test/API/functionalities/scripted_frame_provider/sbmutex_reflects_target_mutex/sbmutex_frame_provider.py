"""
Frame provider whose get_frame_at_index confirms SBMutex aliases the target's
genuine API mutex rather than the no-op TargetAPIMutex resolves to under the
can_bypass_target_api_mutex policy that ScriptedPythonInterface pushes for a
callback's entire duration.

Every acquisition here uses try_lock() and runs on a freshly spawned thread, so
no scripted-extension call is on its stack and none of it is exempt from the
real mutex. Blocking on lock() is never an option: the thread that reaches this
callback may already hold the real mutex and is waiting on this code, so a
blocking acquisition on any thread deadlocks the session.

TargetAPIMutex::try_lock() returns true unconditionally when it resolves to the
no-op, so any *failed* try_lock() proves the handle reached the real mutex.
"""

import threading

from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider

# A second handle could not take the mutex the first one holds, so the two alias
# the same real mutex.
CONTENDED = "second handle contended with the first"
# Some other thread already held the real mutex, which a no-op cannot do.
OTHER_HOLDER = "another thread already held the real mutex"
# Failure: two handles held the mutex at once, so at least one is a no-op.
UNCONTENDED = "two handles held the real mutex at once"


class ContentionCheckFrameProvider(ScriptedFrameProvider):
    @staticmethod
    def get_description():
        return "Provider that checks SBMutex contention from background threads"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)
        self.artifact_path = None
        if self.args is not None:
            value = self.args.GetValueForKey("artifact_path")
            if value.IsValid():
                self.artifact_path = value.GetStringValue(4096)

    def _check_contention(self):
        first = self.target.GetAPIMutex()
        if not first.try_lock():
            self._record(OTHER_HOLDER)
            return

        outcome = [UNCONTENDED]

        def check_from_another_thread():
            other = self.target.GetAPIMutex()
            if other.try_lock():
                other.unlock()
            else:
                outcome[0] = CONTENDED

        other_thread = threading.Thread(target=check_from_another_thread)
        other_thread.start()
        other_thread.join()
        first.unlock()
        self._record(outcome[0])

    def _record(self, outcome):
        with open(self.artifact_path, "a") as f:
            f.write(outcome + "\n")

    def get_frame_at_index(self, index):
        if index >= len(self.input_frames):
            return None

        if index == 0 and self.artifact_path:
            # Obtaining a handle locks nothing, so it is safe on this thread;
            # only the try_lock() calls have to run elsewhere. Every spawned
            # thread is joined before returning, so nothing holds the mutex
            # once the bypass ends.
            checker = threading.Thread(target=self._check_contention)
            checker.start()
            checker.join()

        frame = self.input_frames[index]
        if frame is None:
            return None
        return {"idx": index, "pc": frame.GetPC()}
