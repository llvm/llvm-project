from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

from fork_testbase import GdbRemoteForkTestBase


class TestGdbRemoteForkNonStop(GdbRemoteForkTestBase):
    @add_test_categories(["fork"])
    def test_vfork_nonstop(self):
        parent_pid, parent_tid = self.fork_and_detach_test("vfork",
                                                           nonstop=True)

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            "send packet: $OK#00",
            {"direction": "send",
             "regex": r"%Stop:T05thread:p{}[.]{}.*vforkdone.*".format(
                 parent_pid, parent_tid),
             },
            "read packet: $vStopped#00",
            "send packet: $OK#00",
            "read packet: $c#00",
            "send packet: $OK#00",
            "send packet: %Stop:W00;process:{}#00".format(parent_pid),
            "read packet: $vStopped#00",
            "send packet: $OK#00",
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_fork_nonstop(self):
        parent_pid, _ = self.fork_and_detach_test("fork", nonstop=True)

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            "send packet: $OK#00",
            "send packet: %Stop:W00;process:{}#00".format(parent_pid),
            "read packet: $vStopped#00",
            "send packet: $OK#00",
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_fork_follow_nonstop(self):
        self.fork_and_follow_test("fork", nonstop=True)

    @add_test_categories(["fork"])
    def test_vfork_follow_nonstop(self):
        self.fork_and_follow_test("vfork", nonstop=True)

    @add_test_categories(["fork"])
    def test_detach_all_nonstop(self):
        self.detach_all_test(nonstop=True)

    @add_test_categories(["fork"])
    def test_kill_all_nonstop(self):
        parent_pid, _, child_pid, _ = self.start_fork_test(["fork"],
                                                           nonstop=True)

        exit_regex = "X09;process:([0-9a-f]+)"
        # Depending on a potential race, the second kill may make it into
        # the async queue before we issue vStopped or after.  In the former
        # case, we should expect the exit status in reply to vStopped.
        # In the latter, we should expect an OK response (queue empty),
        # followed by another async notification.
        vstop_regex = "[$](OK|{})#.*".format(exit_regex)
        self.test_sequence.add_log_lines([
            # kill all processes
            "read packet: $k#00",
            "send packet: $OK#00",
            {"direction": "send", "regex": "%Stop:{}#.*".format(exit_regex),
             "capture": {1: "pid1"}},
            "read packet: $vStopped#00",
            {"direction": "send", "regex": vstop_regex,
             "capture": {1: "vstop_reply", 2: "pid2"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid1 = ret["pid1"]
        if ret["vstop_reply"] == "OK":
            self.reset_test_sequence()
            self.test_sequence.add_log_lines([
                {"direction": "send", "regex": "%Stop:{}#.*".format(exit_regex),
                 "capture": {1: "pid2"}},
            ], True)
            ret = self.expect_gdbremote_sequence()
        pid2 = ret["pid2"]
        self.reset_test_sequence()
        self.test_sequence.add_log_lines([
            "read packet: $vStopped#00",
            "send packet: $OK#00",
        ], True)
        self.expect_gdbremote_sequence()
        self.assertEqual(set([pid1, pid2]), set([parent_pid, child_pid]))

    @add_test_categories(["fork"])
    def test_vkill_both_nonstop(self):
        self.vkill_test(kill_parent=True, kill_child=True, nonstop=True)

    @add_test_categories(["fork"])
    def test_c_interspersed_nonstop(self):
        self.resume_one_test(run_order=["parent", "child", "parent", "child"],
                             nonstop=True)

    @add_test_categories(["fork"])
    def test_vCont_interspersed_nonstop(self):
        self.resume_one_test(run_order=["parent", "child", "parent", "child"],
                             use_vCont=True, nonstop=True)
