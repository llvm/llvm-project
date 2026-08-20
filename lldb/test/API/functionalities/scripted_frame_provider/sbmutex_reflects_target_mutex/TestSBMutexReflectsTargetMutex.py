"""
Test that a scripted frame provider calling SBTarget.GetAPIMutex() from
get_frame_at_index gets a handle that reflects the state of the target's
real, shared API mutex, even though the callback's own thread is exempt
from having to serialize on it. SBMutex is meant to be obtainable inside
a bypassed scripted callback and locked later, once that bypass no
longer applies (e.g. on a different thread with no scripted-extension
call on its stack, as this test's own provider does; see
sbmutex_frame_provider.py). So it must always alias the genuine target
mutex rather than resolving to the no-op the bypass policy makes it for
internal callers.

The provider obtains the mutex from inside get_frame_at_index, which is
safe since obtaining a handle doesn't resolve or lock anything, and does
every acquisition with try_lock() on threads it spawns and joins (see
sbmutex_frame_provider.py). Nothing may block on lock() there, on any
thread: the thread that reaches the callback may already hold the real
mutex while waiting on the provider, so a blocking acquisition deadlocks
the session. `bt` is interleaved with `continue` so the callback runs many
times rather than once.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestSBMutexReflectsTargetMutex(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_sbmutex_reflects_target_mutex(self):
        """
        Register a scripted frame provider that checks
        target.GetAPIMutex().try_lock() from get_frame_at_index, then
        repeatedly run `bt` and `continue` through RunCommandInterpreter.
        Should complete without deadlocking, regardless of whether
        contention is observed.
        """
        self.build()

        lldbutil.run_to_name_breakpoint(self, "frame3")

        provider_path = os.path.join(self.getSourceDir(), "sbmutex_frame_provider.py")
        artifact_path = self.getBuildArtifact("contention.txt")
        if os.path.exists(artifact_path):
            os.remove(artifact_path)

        commands = ["command script import " + provider_path]
        commands.append(
            "target frame-provider register "
            "-C sbmutex_frame_provider.ContentionCheckFrameProvider "
            "-k artifact_path -v " + artifact_path
        )
        # `bt` only re-invokes get_frame_at_index when the thread's stack
        # frame list was invalidated by a new stop, so interleave `bt` with
        # `continue` (hitting the same breakpoint again, in a loop in
        # main.c) to get repeated fresh invocations of the check.
        commands.extend(["bt", "continue"] * 20)
        commands.append("quit")

        stdin_path = self.getBuildArtifact("stdin.txt")
        stdout_path = self.getBuildArtifact("stdout.txt")
        with open(stdin_path, "w") as f:
            f.write("\n".join(commands) + "\n")

        with open(stdin_path, "r") as in_fileH, open(stdout_path, "w") as out_fileH:
            in_sbf = lldb.SBFile(in_fileH.fileno(), "r", False)
            out_sbf = lldb.SBFile(out_fileH.fileno(), "w", False)
            self.assertSuccess(self.dbg.SetInputFile(in_sbf))
            self.assertSuccess(self.dbg.SetOutputFile(out_sbf))
            self.assertSuccess(self.dbg.SetErrorFile(out_sbf))

            options = lldb.SBCommandInterpreterRunOptions()
            options.SetEchoCommands(False)
            options.SetPrintResults(True)
            options.SetStopOnError(False)
            options.SetStopOnCrash(False)

            n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, options, 0, False, False
            )

        with open(stdout_path, "r") as out_fileH:
            output = out_fileH.read()

        self.assertFalse(has_crashed, "lldb should not have crashed")
        self.assertTrue(quit_requested, "quit command should have been processed")
        self.assertEqual(n_errors, 0, f"unexpected errors in output:\n{output}")
        self.assertIn("successfully registered scripted frame provider", output)

        self.assertTrue(
            os.path.exists(artifact_path),
            "get_frame_at_index should have run and recorded at least one outcome",
        )
        with open(artifact_path, "r") as f:
            outcomes = [line.strip() for line in f if line.strip()]

        self.assertTrue(outcomes, "expected at least one recorded outcome")
        # Either recorded outcome means a try_lock() failed, which a no-op
        # handle can never do. Only the third outcome, two handles holding the
        # mutex at once, indicates SBMutex resolved to the bypass no-op.
        self.assertTrue(
            set(outcomes)
            <= {
                "second handle contended with the first",
                "another thread already held the real mutex",
            },
            f"SBMutex did not alias the real target mutex: {outcomes}",
        )
