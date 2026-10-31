"""
Test that a scripted frame provider can safely call a blocking
SBMutex.lock() from inside get_frame_at_index without deadlocking.

The private state thread can reach this callback without already
holding the target's real API mutex. Without the bypass described
below, a blocking lock() call here could genuinely wait for that mutex,
and deadlock if some other thread (e.g. a `bt` command thread) holds it
at that moment. ScriptedPythonInterface::Dispatch prevents this by
pushing the can_bypass_target_api_mutex policy around the whole
callback. TargetAPIMutex re-checks that policy on every lock() call
rather than caching whatever was current when the SBMutex was
constructed, so lock() here resolves to a genuine no-op instead: no
synchronization primitive is touched at all, and it never contends with
anyone.

This drives a genuine cross-thread race and is best-effort: it raises
the odds of exercising the path within a single invocation but the
important guarantee is that it cannot hang, not that it hits any
particular thread ordering.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestHoldMutexNoDeadlock(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_hold_mutex_no_deadlock(self):
        """
        Register a scripted frame provider that locks and holds
        target.GetAPIMutex() from get_frame_at_index, then run `bt` and
        `continue` through RunCommandInterpreter. Should complete
        without deadlocking.
        """
        self.build()

        lldbutil.run_to_name_breakpoint(self, "frame3")

        provider_path = os.path.join(
            self.getSourceDir(), "hold_mutex_frame_provider.py"
        )
        commands = ["command script import " + provider_path]
        commands.append(
            "target frame-provider register "
            "-C hold_mutex_frame_provider.HoldMutexFrameProvider"
        )
        # Interleave `bt` with `continue` (hitting the same breakpoint
        # again, via a loop in main.c) so get_frame_at_index runs
        # repeatedly instead of once, raising the odds of hitting the
        # race within a single test invocation.
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

            # If the bypass regresses, this call hangs forever (timing
            # out the test run).
            n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, options, 0, False, False
            )

        with open(stdout_path, "r") as out_fileH:
            output = out_fileH.read()

        self.assertFalse(has_crashed, "lldb should not have crashed")
        self.assertTrue(quit_requested, "quit command should have been processed")
        self.assertEqual(n_errors, 0, f"unexpected errors in output:\n{output}")
        self.assertIn("successfully registered scripted frame provider", output)
