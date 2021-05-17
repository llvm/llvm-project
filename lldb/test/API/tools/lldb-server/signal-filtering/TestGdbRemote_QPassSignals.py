# This test makes sure that lldb-server supports and properly handles
# QPassSignals GDB protocol package.

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemote_QPassSignals(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def expect_signal(self, expected_signo):
        self.test_sequence.add_log_lines(["read packet: $vCont;c#a8",
                                          {"direction": "send",
                                           "regex": r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$",
                                           "capture": {1: "hex_exit_code"}},
                                          ],
                                         True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        hex_exit_code = context.get("hex_exit_code")
        self.assertIsNotNone(hex_exit_code)
        self.assertEqual(int(hex_exit_code, 16), expected_signo)

    def expect_exit_code(self, exit_code):
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8",
             "send packet: $W{0:02x}#00".format(exit_code)],
            True)
        self.expect_gdbremote_sequence()


    def ignore_signals(self, signals):
        def signal_name_to_hex(signame):
            return format(lldbutil.get_signal_number(signame), 'x')
        signals_str = ";".join(map(signal_name_to_hex, signals))

        self.test_sequence.add_log_lines(["read packet: $QPassSignals:"
                                          + signals_str + " #00",
                                          "send packet: $OK#00"],
                                         True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipUnlessPlatform(["linux", "android"])
    def test_q_pass_signals(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        expected_signals = ["SIGSEGV",
            "SIGALRM", "SIGFPE", "SIGBUS", "SIGINT", "SIGHUP"]
        signals_to_ignore = ["SIGUSR1", "SIGUSR2"]
        self.ignore_signals(signals_to_ignore)
        for signal_name in expected_signals:
            signo = lldbutil.get_signal_number(signal_name)
            self.expect_signal(signo)
        self.expect_exit_code(len(signals_to_ignore))

    @skipUnlessPlatform(["linux", "android"])
    def test_change_signals_at_runtime(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        expected_signals = ["SIGSEGV", "SIGUSR1", "SIGUSR2",
            "SIGALRM", "SIGHUP"]
        signals_to_ignore = ["SIGFPE", "SIGBUS", "SIGINT"]

        for signal_name in expected_signals:
            signo = lldbutil.get_signal_number(signal_name)
            self.expect_signal(signo)
            if signal_name == "SIGALRM":
                self.ignore_signals(signals_to_ignore)
        self.expect_exit_code(len(signals_to_ignore))

    @skipIfWindows # no signal support
    @expectedFailureNetBSD
    def test_default_signals_behavior(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        expected_signals = ["SIGSEGV", "SIGUSR1", "SIGUSR2",
            "SIGALRM", "SIGFPE", "SIGBUS", "SIGINT", "SIGHUP"]
        for signal_name in expected_signals:
            signo = lldbutil.get_signal_number(signal_name)
            self.expect_signal(signo)
        self.expect_exit_code(0)
