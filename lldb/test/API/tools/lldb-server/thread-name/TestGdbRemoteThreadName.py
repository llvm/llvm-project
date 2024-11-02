import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteThreadName(gdbremote_testcase.GdbRemoteTestCaseBase):

    def run_and_check_name(self, expected_name):
        self.test_sequence.add_log_lines(["read packet: $vCont;c#a8",
                                          {"direction": "send",
                                           "regex":
                                           r"^\$T([0-9a-fA-F]{2})([^#]+)#[0-9a-fA-F]{2}$",
                                           "capture": {
                                               1: "signal",
                                               2: "key_vals_text"}},
                                          ],
                                         True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        sigint = lldbutil.get_signal_number("SIGINT")
        self.assertEqual(sigint, int(context.get("signal"), 16))
        kv_dict = self.parse_key_val_dict(context.get("key_vals_text"))
        self.assertEqual(expected_name, kv_dict.get("name"))

    @skipIfWindows # the test is not updated for Windows.
    def test(self):
        """ Make sure lldb-server can retrieve inferior thread name"""
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        self.run_and_check_name("hello world")
        self.run_and_check_name("goodbye world")
