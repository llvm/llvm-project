"""
Test that a scripted frame provider whose get_frame_at_index touches SB
API (self.input_frames) does not deadlock when running `bt` from the
command interpreter.

GetStoppedExecutionContext (used by SBFrame::IsValid, among others)
unconditionally blocked acquiring the target's API mutex. The command
thread running `bt` already holds that mutex (CommandObjectParsed's
eCommandTryTargetAPILock) and can end up waiting on a StackFrameList
lock held by the debugger's event-handler thread, which is itself
blocked re-acquiring the API mutex from inside this provider's Python
code: an AB-BA deadlock between the command thread and the
event-handler thread.

The event-handler thread only runs when commands are driven through
SBDebugger.RunCommandInterpreter (what the lldb driver itself uses),
not through plain HandleCommand, so this test drives commands that way.

Note: this is a genuine cross-thread race (the command thread vs. the
debugger's event-handler thread), not a deterministic sequential
deadlock, so this test is best-effort: it raises the odds of hitting
the race within a single invocation but cannot guarantee it.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameProviderRegisterCommandAPIMutexDeadlock(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_register_command_then_bt_no_deadlock(self):
        """
        Register a scripted frame provider whose get_frame_at_index
        touches SB API, then repeatedly run `bt` through
        RunCommandInterpreter. Should complete without deadlocking.
        """
        self.build()

        lldbutil.run_to_name_breakpoint(self, "frame3")

        provider_path = os.path.join(self.getSourceDir(), "frame_provider.py")

        commands = ["command script import " + provider_path]
        commands.append(
            "target frame-provider register -C frame_provider.DictFrameProvider"
        )
        # Run `bt` several times to raise the odds of hitting the race
        # (see module docstring).
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

            # If the API-mutex deadlock regresses, this call hangs forever
            # (timing out the test run).
            n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, options, 0, False, False
            )

        with open(stdout_path, "r") as out_fileH:
            output = out_fileH.read()

        self.assertFalse(has_crashed, "lldb should not have crashed")
        self.assertTrue(quit_requested, "quit command should have been processed")
        self.assertEqual(n_errors, 0, f"unexpected errors in output:\n{output}")

        self.assertIn("successfully registered scripted frame provider", output)
