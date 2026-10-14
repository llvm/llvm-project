"""
Test that interrupting a running process on Windows is reported as an
interrupt rather than as an exception.

Windows has no SIGSTOP, so a halt is implemented with DebugBreakProcess(),
which injects a thread into the inferior that runs ntdll!DbgUiRemoteBreakin
and executes an int3. lldb tells that int3 apart from one the inferior itself
executed by the injected thread's entry point; if that stops working, the
interrupt surfaces as a bare 0x80000003 exception instead.

Only lldb-server maps the halt to a signal stop (NativeProcessWindows). The
in-process ProcessWindows plugin has no such mapping: any int3 without a
matching breakpoint site reaches RefreshStateAfterStop's default case and
becomes eStopReasonException, so this test does not apply there.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireWindows
@skipIfWindowsAndNoLLDBServer
class WindowsInterruptTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_interrupt_is_not_reported_as_an_exception(self):
        self.build()
        (target, process, _, _) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.setAsync(True)
        listener = self.dbg.GetListener()

        # Interrupt more than once: the injected thread has to be recognized
        # every time, not just for the first halt.
        for i in range(3):
            process.Continue()
            lldbutil.expect_state_changes(self, listener, process, [lldb.eStateRunning])

            self.assertSuccess(process.Stop(), "interrupt #%d" % i)
            lldbutil.expect_state_changes(self, listener, process, [lldb.eStateStopped])

            for thread in process:
                self.assertNotEqual(
                    thread.GetStopReason(),
                    lldb.eStopReasonException,
                    "interrupt #%d reported as an exception on thread %d: %s"
                    % (i, thread.GetThreadID(), thread.GetStopDescription(256)),
                )

        # The inferior must still run to completion once the loop is let go,
        # i.e. the injected threads did not leave the process wedged.
        self.setAsync(False)
        keep_running = target.FindFirstGlobalVariable("keep_running")
        self.assertTrue(keep_running.IsValid(), "found the loop's exit condition")
        self.assertTrue(keep_running.SetValueFromCString("0"), "cleared keep_running")

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
