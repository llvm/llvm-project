"""
Test Objective-C exception support against gnustep-base's NSException.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGNUstepExceptions(TestBase):
    def test_formatter_in_handler(self):
        """An NSException formats by its reason, from runtime metadata.

        gnustep-base declares _e_name and _e_reason behind
        GS_EXPOSE(NSException), so nothing here has debug info for them.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break in handler", lldb.SBFileSpec("main.m")
        )
        self.runCmd("settings set target.prefer-dynamic-value run-target")
        self.expect(
            "frame variable -d run-target caught",
            substrs=['@"the bad thing happened"'],
        )
        self.expect("po caught", substrs=["the bad thing happened"])

    def test_stops_at_throw(self):
        """`breakpoint set -E objc` stops where the exception is raised, and
        the frame recognizer presents the thrown object."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.runCmd("breakpoint set -E objc")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread, "stopped on the exception breakpoint")
        # The recognizer's description surfaces through the command layer
        # rather than through SBThread::GetStopDescription.
        self.expect(
            "thread list",
            substrs=["stopped", "stop reason = hit Objective-C exception"],
        )

        # The recognizer synthesizes `exception` as an argument, so it is
        # visible even though objc_exception_throw has no debug info.
        exception = thread.GetCurrentException()
        self.assertTrue(exception.IsValid(), "thread has a current exception")
        self.assertIn("NSException", exception.GetTypeName())

    def test_exception_backtrace(self):
        """`thread exception` reports the stack recorded inside the exception.

        gnustep-base captures it on the first -[NSException raise], into a
        GSStackTrace whose `returns` array holds raw return addresses - so this
        is a pure memory read, no NSArray walk and no inferior call."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.runCmd("breakpoint set -E objc")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread, "stopped on the exception breakpoint")

        history = thread.GetCurrentExceptionBacktrace()
        self.assertTrue(history.IsValid(), "exception carries a backtrace")

        # The frames that raised must be present and attributed to this file.
        # Frames below them are gnustep-base's own capture machinery, which is
        # deliberately not filtered out - see GetBacktraceThreadFromException.
        names = [f.GetFunctionName() for f in history.frames]
        for expected in ["raiser", "main"]:
            self.assertEqual(
                len([n for n in names if n == expected]),
                1,
                f"exactly one {expected} frame in {names}",
            )

        # The recorded stack must be the one captured at the raise, not the
        # thread's current frames - which at the throw site also contain
        # exactly one `raiser` and one `main`, so the check above cannot tell
        # them apart on its own. gnustep-base's own capture machinery is only
        # ever present in the recorded stack.
        self.assertTrue(
            any(n and "GSStackTrace" in n for n in names),
            f"recorded stack, not the live one: {names}",
        )

        # The command layer is the user-visible surface for this.
        self.expect("thread exception", substrs=["-[NSException raise]"])

    def test_exception_is_recognized_argument(self):
        """The thrown object is exposed as a recognized argument, which is
        what lldb-dap's exception view reads."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.runCmd("breakpoint set -E objc")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        options = lldb.SBVariablesOptions()
        options.SetIncludeArguments(False)
        options.SetIncludeRecognizedArguments(True)
        options.SetIncludeLocals(False)
        options.SetIncludeStatics(False)
        variables = thread.GetFrameAtIndex(0).GetVariables(options)
        self.assertEqual(variables.GetSize(), 1)
        self.assertEqual(variables.GetValueAtIndex(0).GetName(), "exception")
        self.assertEqual(
            variables.GetValueAtIndex(0).GetValueType(),
            lldb.eValueTypeVariableArgument,
        )
