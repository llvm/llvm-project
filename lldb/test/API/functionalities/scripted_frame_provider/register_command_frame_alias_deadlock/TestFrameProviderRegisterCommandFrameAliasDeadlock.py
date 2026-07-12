"""
Test that a scripted frame provider that forwards frames under their own
index (identity forwarding) does not deadlock when running `bt` from the
command interpreter.

ScriptedFrameProvider::GetFrameAtIndex reused the parent list's live
StackFrame object directly whenever a provider forwarded a frame under
its own index, instead of wrapping it in a BorrowedStackFrame. Frame
construction unconditionally re-tags the returned frame as belonging to
the child list, corrupting the parent list's cached frame. Once that
frame's corrupted list identity is later resolved, it points back at
the (possibly still-being-built) child list, and a thread already
holding that list's writer lock can self-deadlock taking the reader
lock.

The event-handler thread that can trigger this only runs when commands
are driven through SBDebugger.RunCommandInterpreter (what the lldb
driver itself uses), not plain HandleCommand, so this test drives
commands that way.

Note: this is a genuine cross-thread race (the command thread vs. the
debugger's event-handler thread), not a deterministic sequential
deadlock, so this test is best-effort -- like the sibling
runlock_reentrant_deadlock/was_hit_deadlock tests, it raises the odds of
hitting the race within a single invocation but cannot guarantee it.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameProviderRegisterCommandFrameAliasDeadlock(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_register_command_then_bt_no_deadlock(self):
        """
        Register a scripted frame provider that identity-forwards every
        frame, then repeatedly run `bt` through RunCommandInterpreter.
        Should complete without deadlocking.
        """
        self.build()

        lldbutil.run_to_name_breakpoint(self, "frame3")

        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")

        commands = ["command script import " + provider_path]
        commands.append(
            "target frame-provider register -C frame_provider.IdentityProvider"
        )
        # Run `bt` several times to raise the odds of hitting the race
        # between the command thread and the debugger's event-handler
        # thread within a single test invocation.
        commands.extend(["bt"] * 20)
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

            # If the frame-aliasing self-deadlock regresses, this call
            # hangs forever (timing out the test run).
            n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, options, 0, False, False
            )

        with open(stdout_path, "r") as out_fileH:
            output = out_fileH.read()

        self.assertFalse(has_crashed, "lldb should not have crashed")
        self.assertTrue(quit_requested, "quit command should have been processed")
        self.assertEqual(n_errors, 0, f"unexpected errors in output:\n{output}")

        self.assertIn("successfully registered scripted frame provider", output)
        self.assertIn("frame3", output)
        self.assertIn("frame2", output)
        self.assertIn("frame1", output)
