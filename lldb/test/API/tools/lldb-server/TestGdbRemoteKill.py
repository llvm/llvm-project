

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteKill(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet

    def attach_commandline_kill_after_initial_stop(self):
        reg_expr = r"^\$[XW][0-9a-fA-F]+([^#]*)#[0-9A-Fa-f]{2}"
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines([
            "read packet: $k#6b",
            {"direction": "send", "regex": reg_expr},
        ], True)

        if self.stub_sends_two_stop_notifications_on_kill:
            # Add an expectation for a second X result for stubs that send two
            # of these.
            self.test_sequence.add_log_lines([
                {"direction": "send", "regex": reg_expr},
            ], True)

        self.expect_gdbremote_sequence()

        # Wait a moment for completed and now-detached inferior process to
        # clear.
        time.sleep(self.DEFAULT_SLEEP)

        if not lldb.remote_platform:
            # Process should be dead now. Reap results.
            poll_result = procs["inferior"].poll()
            self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not
        # running.
        self.assertFalse(
            lldbgdbserverutils.process_is_running(
                procs["inferior"].pid, False))

    @debugserver_test
    def test_attach_commandline_kill_after_initial_stop_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_attach()
        self.attach_commandline_kill_after_initial_stop()

    @llgs_test
    def test_attach_commandline_kill_after_initial_stop_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_attach()
        self.attach_commandline_kill_after_initial_stop()
