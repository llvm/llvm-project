import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteFork(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def fork_and_detach_test(self, variant):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=[variant])
        self.add_qSupported_packets(["multiprocess+",
                                     "{}-events+".format(variant)])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("{}-events+".format(variant), ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        fork_regex = "[$]T.*;{}:p([0-9a-f]*)[.]([0-9a-f]*).*".format(variant)
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": fork_regex,
             "capture": {1: "pid", 2: "tid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid = int(ret["pid"], 16)
        self.reset_test_sequence()

        # detach the forked child
        self.test_sequence.add_log_lines([
            "read packet: $D;{:x}#00".format(pid),
            {"direction": "send", "regex": r"[$]OK#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()
        self.reset_test_sequence()

    @add_test_categories(["fork"])
    def test_fork(self):
        self.fork_and_detach_test("fork")

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": r"[$]W00#.*"},
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vfork(self):
        self.fork_and_detach_test("vfork")

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": r"[$]T.*vforkdone.*"},
            "read packet: $c#00",
            {"direction": "send", "regex": r"[$]W00#.*"},
        ], True)
        self.expect_gdbremote_sequence()
