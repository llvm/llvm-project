"""
Test that a scripted frame provider calling SBTarget.GetAPIMutex() from
get_frame_at_index gets a handle that reflects the state of the target's
real, shared API mutex, even though the callback's own thread is exempt
from having to serialize on it.

SBMutex is meant to be obtainable inside a bypassed scripted callback and
locked later, once that bypass no longer applies -- e.g. on a different
thread with no scripted-extension call on its stack, as this test's own
provider does (see sbmutex_frame_provider.py). So it must always alias
the genuine target mutex rather than resolving to the no-op the bypass
policy makes it for internal callers. This test drives the same
kind of command-thread / internal-thread race as
TestFrameProviderRegisterCommandAPIMutexDeadlock, interleaving `bt` with
`continue` (hitting the same breakpoint again each time, via a loop in
main.c) so get_frame_at_index runs many times instead of once.

The provider obtains the mutex from inside get_frame_at_index (safe --
obtaining a handle doesn't resolve or lock anything), but the actual
try_lock() runs on a plain background thread it spawns for the check (see
sbmutex_frame_provider.py for why: ScriptedPythonInterface::Dispatch pushes
the API-mutex bypass policy for the callback's entire duration, so
try_lock() on the callback's own thread would always resolve to a genuine
no-op -- no synchronization primitive touched at all, so no other thread
could ever contend on it, making the check meaningless regardless of what
any other thread is doing to the real mutex). The background thread never
had that policy pushed, so its try_lock() resolves to the real, shared
mutex: if some other thread (e.g. the command thread running `bt`) happens
to hold it at that moment, this correctly observes it as contended.

Only try_lock() is used, which never blocks, so this cannot deadlock
regardless of the outcome. An earlier version of this test tried to
widen the race window by having the callback actually lock() and hold
the mutex for a short duration, on the assumption that whichever thread
reaches this callback already holds the real mutex first. That
assumption is wrong -- LLDB's private state thread can reach this
callback without already holding it -- so that held mutex.lock() call
could genuinely block, and reproducibly deadlocked in practice. Do not
reintroduce a blocking acquisition here.

Observing contention is a genuine cross-thread race, so -- like the
sibling runlock_reentrant_deadlock/was_hit_deadlock/
register_command_api_mutex_deadlock tests -- this is best-effort: it
raises the odds of witnessing it within a single invocation but cannot
guarantee it, and the test does not require it to pass.
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
        # main.c) to get repeated fresh invocations, raising the odds of
        # hitting the race within a single test invocation.
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
        self.assertTrue(
            all(
                o
                in (
                    "another thread held the real target API mutex",
                    "no other thread held the real target API mutex",
                )
                for o in outcomes
            ),
            f"unexpected outcome values: {outcomes}",
        )
        # Whether this specific outcome occurs is a race (see module
        # docstring); not asserted on here.
