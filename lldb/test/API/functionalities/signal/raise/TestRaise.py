"""Test that we handle inferiors that send signals to themselves"""


import lldb
import re
from lldbsuite.test.lldbplatformutil import getDarwinOSTriples
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWindows  # signals do not exist on Windows
class RaiseTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNetBSD  # Hangs on NetBSD
    def test_sigstop(self):
        self.build()
        self.signal_test("SIGSTOP", False)
        # passing of SIGSTOP is not correctly handled, so not testing that
        # scenario: https://llvm.org/bugs/show_bug.cgi?id=23574

    @skipIfDarwin  # darwin does not support real time signals
    @skipIfTargetAndroid()
    def test_sigsigrtmin(self):
        self.build()
        self.signal_test("SIGRTMIN", True)

    @skipIfNetBSD  # Hangs on NetBSD
    def test_sigtrap(self):
        self.build()
        self.signal_test("SIGTRAP", True)

    def launch(self, target, signal):
        # launch the process, do not stop at entry point.
        # If we have gotten the default for this signal, reset that as well.
        if len(self.default_pass) != 0:
            lldbutil.set_actions_for_signal(
                self, signal, self.default_pass, self.default_stop, self.default_notify
            )

        process = target.LaunchSimple(
            [signal], None, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(), "Thread should be stopped due to a breakpoint"
        )
        return process

    def set_handle(self, signal, pass_signal, stop_at_signal, notify_signal):
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle %s -p %s -s %s -n %s"
            % (signal, pass_signal, stop_at_signal, notify_signal),
            return_obj,
        )
        self.assertTrue(return_obj.Succeeded(), "Setting signal handling failed")

    def signal_test(self, signal, test_passing):
        """Test that we handle inferior raising signals"""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        lldbutil.run_break_set_by_symbol(self, "main")
        self.default_pass = ""
        self.default_stop = ""
        self.default_notify = ""

        # launch
        process = self.launch(target, signal)
        signo = process.GetUnixSignals().GetSignalNumberFromName(signal)

        # retrieve default signal disposition
        (
            self.default_pass,
            self.default_stop,
            self.default_notify,
        ) = lldbutil.get_actions_for_signal(self, signal)

        # Make sure we stop at the signal
        lldbutil.set_actions_for_signal(self, signal, "false", "true", "true")
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertGreaterEqual(
            thread.GetStopReasonDataCount(), 1, "There was data in the event."
        )
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0), signo, "The stop signal was %s" % signal
        )

        # Continue until we exit.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the
        # notification.
        lldbutil.set_actions_for_signal(self, signal, "false", "false", "true")
        self.expect("process continue", substrs=["stopped and restarted", signal])
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the
        # notification.
        lldbutil.set_actions_for_signal(self, signal, "false", "false", "false")
        self.expect(
            "process continue", substrs=["stopped and restarted"], matching=False
        )
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        if not test_passing:
            # reset signal handling to default
            lldbutil.set_actions_for_signal(
                self, signal, self.default_pass, self.default_stop, self.default_notify
            )
            return

        # launch again
        process = self.launch(target, signal)

        # Make sure we stop at the signal
        lldbutil.set_actions_for_signal(self, signal, "true", "true", "true")
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertGreaterEqual(
            thread.GetStopReasonDataCount(), 1, "There was data in the event."
        )
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            process.GetUnixSignals().GetSignalNumberFromName(signal),
            "The stop signal was %s" % signal,
        )

        # Continue until we exit. The process should receive the signal.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the notification. Process
        # should receive the signal.
        lldbutil.set_actions_for_signal(self, signal, "true", "false", "true")
        self.expect("process continue", substrs=["stopped and restarted", signal])
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the notification. Process
        # should receive the signal.
        lldbutil.set_actions_for_signal(self, signal, "true", "false", "false")
        self.expect(
            "process continue", substrs=["stopped and restarted"], matching=False
        )
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # reset signal handling to default
        lldbutil.set_actions_for_signal(
            self, signal, self.default_pass, self.default_stop, self.default_notify
        )
