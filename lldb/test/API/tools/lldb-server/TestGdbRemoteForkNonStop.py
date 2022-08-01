from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

from fork_testbase import GdbRemoteForkTestBase


class TestGdbRemoteForkNonStop(GdbRemoteForkTestBase):
    def setUp(self):
        GdbRemoteForkTestBase.setUp(self)
        if self.getPlatform() == "linux" and self.getArchitecture() in ['arm', 'aarch64']:
            self.skipTest("Unsupported for Arm/AArch64 Linux")

    @add_test_categories(["fork"])
    def test_vfork_nonstop(self):
        parent_pid, parent_tid = self.fork_and_detach_test("vfork",
                                                           nonstop=True)

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            "send packet: $OK#00",
            {"direction": "send",
             "regex": r"%Stop:T[0-9a-fA-F]{{2}}thread:p{}[.]{}.*vforkdone.*"
                      .format(parent_pid, parent_tid),
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

    def get_all_output_via_vStdio(self, output_test):
        # The output may be split into an arbitrary number of messages.
        # Loop until we have everything. The first message is waiting for us
        # in the packet queue.
        output = self._server.get_raw_output_packet()
        while not output_test(output):
            self._server.send_packet(b"vStdio")
            output += self._server.get_raw_output_packet()
        return output

    @add_test_categories(["fork"])
    def test_c_both_nonstop(self):
        lock1 = self.getBuildArtifact("lock1")
        lock2 = self.getBuildArtifact("lock2")
        parent_pid, parent_tid, child_pid, child_tid = (
            self.start_fork_test(["fork", "process:sync:" + lock1, "print-pid",
                                  "process:sync:" + lock2, "stop"],
                                 nonstop=True))

        self.test_sequence.add_log_lines([
            "read packet: $Hcp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: $OK#00",
            "read packet: $c#00",
            "send packet: $OK#00",
            "read packet: $Hcp{}.{}#00".format(child_pid, child_tid),
            "send packet: $OK#00",
            "read packet: $c#00",
            "send packet: $OK#00",
            {"direction": "send", "regex": "%Stop:T.*"},
            ], True)
        self.expect_gdbremote_sequence()

        output = self.get_all_output_via_vStdio(
            lambda output: output.count(b"PID: ") >= 2)
        self.assertEqual(output.count(b"PID: "), 2)
        self.assertIn("PID: {}".format(int(parent_pid, 16)).encode(), output)
        self.assertIn("PID: {}".format(int(child_pid, 16)).encode(), output)

    @add_test_categories(["fork"])
    def test_vCont_both_nonstop(self):
        lock1 = self.getBuildArtifact("lock1")
        lock2 = self.getBuildArtifact("lock2")
        parent_pid, parent_tid, child_pid, child_tid = (
            self.start_fork_test(["fork", "process:sync:" + lock1, "print-pid",
                                  "process:sync:" + lock2, "stop"],
                                 nonstop=True))

        self.test_sequence.add_log_lines([
            "read packet: $vCont;c:p{}.{};c:p{}.{}#00".format(
                parent_pid, parent_tid, child_pid, child_tid),
            "send packet: $OK#00",
            {"direction": "send", "regex": "%Stop:T.*"},
            ], True)
        self.expect_gdbremote_sequence()

        output = self.get_all_output_via_vStdio(
            lambda output: output.count(b"PID: ") >= 2)
        self.assertEqual(output.count(b"PID: "), 2)
        self.assertIn("PID: {}".format(int(parent_pid, 16)).encode(), output)
        self.assertIn("PID: {}".format(int(child_pid, 16)).encode(), output)

    def vCont_both_nonstop_test(self, vCont_packet):
        lock1 = self.getBuildArtifact("lock1")
        lock2 = self.getBuildArtifact("lock2")
        parent_pid, parent_tid, child_pid, child_tid = (
            self.start_fork_test(["fork", "process:sync:" + lock1, "print-pid",
                                  "process:sync:" + lock2, "stop"],
                                 nonstop=True))

        self.test_sequence.add_log_lines([
            "read packet: ${}#00".format(vCont_packet),
            "send packet: $OK#00",
            {"direction": "send", "regex": "%Stop:T.*"},
            ], True)
        self.expect_gdbremote_sequence()

        output = self.get_all_output_via_vStdio(
            lambda output: output.count(b"PID: ") >= 2)
        self.assertEqual(output.count(b"PID: "), 2)
        self.assertIn("PID: {}".format(int(parent_pid, 16)).encode(), output)
        self.assertIn("PID: {}".format(int(child_pid, 16)).encode(), output)

    @add_test_categories(["fork"])
    def test_vCont_both_implicit_nonstop(self):
        self.vCont_both_nonstop_test("vCont;c")

    @add_test_categories(["fork"])
    def test_vCont_both_minus_one_nonstop(self):
        self.vCont_both_nonstop_test("vCont;c:p-1.-1")
