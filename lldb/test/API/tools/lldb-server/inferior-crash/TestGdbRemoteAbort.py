import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteAbort(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows # No signal is sent on Windows.
    # std::abort() on <= API 16 raises SIGSEGV - b.android.com/179836
    @expectedFailureAndroid(api_levels=list(range(16 + 1)))
    def test_inferior_abort_received_llgs(self):
        self.build()

        procs = self.prep_debug_monitor_and_inferior(inferior_args=["abort"])
        self.assertIsNotNone(procs)

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
        self.assertEqual(int(hex_exit_code, 16),
                         lldbutil.get_signal_number('SIGABRT'))
