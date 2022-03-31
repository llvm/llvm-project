import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteFork(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["fork"])
    def test_fork_multithreaded(self):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["thread:new"]*2 + ["fork"])
        self.add_qSupported_packets(["multiprocess+", "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        fork_regex = "[$]T.*;fork:p([0-9a-f]+)[.]([0-9a-f]+).*"
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

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $k#00",
        ], True)
        self.expect_gdbremote_sequence()

    def fork_and_detach_test(self, variant):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=[variant])
        self.add_qSupported_packets(["multiprocess+",
                                     "{}-events+".format(variant)])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("{}-events+".format(variant), ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        fork_regex = "[$]T.*;{}:p([0-9a-f]+)[.]([0-9a-f]+).*".format(variant)
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

    def fork_and_follow_test(self, variant):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=[variant])
        self.add_qSupported_packets(["multiprocess+",
                                     "{}-events+".format(variant)])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("{}-events+".format(variant), ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        procinfo_regex = "[$]pid:([0-9a-f]+);.*"
        fork_regex = "[$]T.*;{}:p([0-9a-f]+)[.]([0-9a-f]+).*".format(variant)
        self.test_sequence.add_log_lines([
            "read packet: $qProcessInfo#00",
            {"direction": "send", "regex": procinfo_regex,
             "capture": {1: "parent_pid"}},
            "read packet: $c#00",
            {"direction": "send", "regex": fork_regex,
             "capture": {1: "pid", 2: "tid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid, pid, tid = (int(ret[x], 16) for x
                                in ("parent_pid", "pid", "tid"))
        self.reset_test_sequence()

        # switch to the forked child
        self.test_sequence.add_log_lines([
            "read packet: $Hgp{:x}.{:x}#00".format(pid, tid),
            {"direction": "send", "regex": r"[$]OK#.*"},
            "read packet: $Hcp{:x}.{:x}#00".format(pid, tid),
            {"direction": "send", "regex": r"[$]OK#.*"},
        ], True)

        # detach the parent
        self.test_sequence.add_log_lines([
            "read packet: $D;{:x}#00".format(parent_pid),
            {"direction": "send", "regex": r"[$]OK#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()
        self.reset_test_sequence()

        # resume the child
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": r"[$]W00#.*"},
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_fork_follow(self):
        self.fork_and_follow_test("fork")

    @add_test_categories(["fork"])
    def test_vfork_follow(self):
        self.fork_and_follow_test("vfork")

    @add_test_categories(["fork"])
    def test_select_wrong_pid(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets(["multiprocess+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("multiprocess+", ret["qSupported_response"])
        self.reset_test_sequence()

        # get process pid
        procinfo_regex = "[$]pid:([0-9a-f]+);.*"
        self.test_sequence.add_log_lines([
            "read packet: $qProcessInfo#00",
            {"direction": "send", "regex": procinfo_regex,
             "capture": {1: "pid"}},
            "read packet: $qC#00",
            {"direction": "send", "regex": "[$]QC([0-9a-f]+)#.*",
             "capture": {1: "tid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid, tid = (int(ret[x], 16) for x in ("pid", "tid"))
        self.reset_test_sequence()

        # try switching to correct pid
        self.test_sequence.add_log_lines([
            "read packet: $Hgp{:x}.{:x}#00".format(pid, tid),
            {"direction": "send", "regex": r"[$]OK#.*"},
            "read packet: $Hcp{:x}.{:x}#00".format(pid, tid),
            {"direction": "send", "regex": r"[$]OK#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()

        # try switching to invalid tid
        self.test_sequence.add_log_lines([
            "read packet: $Hgp{:x}.{:x}#00".format(pid, tid+1),
            {"direction": "send", "regex": r"[$]E15#.*"},
            "read packet: $Hcp{:x}.{:x}#00".format(pid, tid+1),
            {"direction": "send", "regex": r"[$]E15#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()

        # try switching to invalid pid
        self.test_sequence.add_log_lines([
            "read packet: $Hgp{:x}.{:x}#00".format(pid+1, tid),
            {"direction": "send", "regex": r"[$]Eff#.*"},
            "read packet: $Hcp{:x}.{:x}#00".format(pid+1, tid),
            {"direction": "send", "regex": r"[$]Eff#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_detach_current(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets(["multiprocess+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("multiprocess+", ret["qSupported_response"])
        self.reset_test_sequence()

        # get process pid
        procinfo_regex = "[$]pid:([0-9a-f]+);.*"
        self.test_sequence.add_log_lines([
            "read packet: $qProcessInfo#00",
            {"direction": "send", "regex": procinfo_regex,
             "capture": {1: "pid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid = int(ret["pid"], 16)
        self.reset_test_sequence()

        # detach the process
        self.test_sequence.add_log_lines([
            "read packet: $D;{:x}#00".format(pid),
            {"direction": "send", "regex": r"[$]OK#.*"},
            "read packet: $qC#00",
            {"direction": "send", "regex": r"[$]E44#.*"},
        ], True)
        ret = self.expect_gdbremote_sequence()
