import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteFork(gdbremote_testcase.GdbRemoteTestCaseBase):

    fork_regex = ("[$]T05thread:p([0-9a-f]+)[.]([0-9a-f]+);.*"
                  "{}:p([0-9a-f]+)[.]([0-9a-f]+).*")
    fork_capture = {1: "parent_pid", 2: "parent_tid",
                    3: "child_pid", 4: "child_tid"}
    procinfo_regex = "[$]pid:([0-9a-f]+);.*"

    @add_test_categories(["fork"])
    def test_fork_multithreaded(self):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["thread:new"]*2 + ["fork"])
        self.add_qSupported_packets(["multiprocess+", "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format("fork"),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        child_pid = ret["child_pid"]
        self.reset_test_sequence()

        # detach the forked child
        self.test_sequence.add_log_lines([
            "read packet: $D;{}#00".format(child_pid),
            "send packet: $OK#00",
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
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format(variant),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        parent_tid = ret["parent_tid"]
        child_pid = ret["child_pid"]
        child_tid = ret["child_tid"]
        self.reset_test_sequence()

        # detach the forked child
        self.test_sequence.add_log_lines([
            "read packet: $D;{}#00".format(child_pid),
            "send packet: $OK#00",
            # verify that the current process is correct
            "read packet: $qC#00",
            "send packet: $QC{}#00".format(parent_tid),
            # verify that the correct processes are detached/available
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: $Eff#00",
            "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: $OK#00",
        ], True)
        self.expect_gdbremote_sequence()
        self.reset_test_sequence()
        return parent_pid, parent_tid

    @add_test_categories(["fork"])
    def test_fork(self):
        parent_pid, _ = self.fork_and_detach_test("fork")

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            "send packet: $W00;process:{}#00".format(parent_pid),
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vfork(self):
        parent_pid, parent_tid = self.fork_and_detach_test("vfork")

        # resume the parent
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send",
             "regex": r"[$]T05thread:p{}[.]{}.*vforkdone.*".format(parent_pid,
                                                                   parent_tid),
             },
            "read packet: $c#00",
            "send packet: $W00;process:{}#00".format(parent_pid),
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
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format(variant),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        parent_tid = ret["parent_tid"]
        child_pid = ret["child_pid"]
        child_tid = ret["child_tid"]
        self.reset_test_sequence()

        # switch to the forked child
        self.test_sequence.add_log_lines([
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: $OK#00",
            "read packet: $Hcp{}.{}#00".format(child_pid, child_tid),
            "send packet: $OK#00",
            # detach the parent
            "read packet: $D;{}#00".format(parent_pid),
            "send packet: $OK#00",
            # verify that the correct processes are detached/available
            "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: $Eff#00",
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: $OK#00",
            # then resume the child
            "read packet: $c#00",
            "send packet: $W00;process:{}#00".format(child_pid),
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
        self.test_sequence.add_log_lines([
            "read packet: $qProcessInfo#00",
            {"direction": "send", "regex": self.procinfo_regex,
             "capture": {1: "pid"}},
            "read packet: $qC#00",
            {"direction": "send", "regex": "[$]QC([0-9a-f]+)#.*",
             "capture": {1: "tid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid, tid = (int(ret[x], 16) for x in ("pid", "tid"))
        self.reset_test_sequence()

        self.test_sequence.add_log_lines([
            # try switching to correct pid
            "read packet: $Hgp{:x}.{:x}#00".format(pid, tid),
            "send packet: $OK#00",
            "read packet: $Hcp{:x}.{:x}#00".format(pid, tid),
            "send packet: $OK#00",
            # try switching to invalid tid
            "read packet: $Hgp{:x}.{:x}#00".format(pid, tid+1),
            "send packet: $E15#00",
            "read packet: $Hcp{:x}.{:x}#00".format(pid, tid+1),
            "send packet: $E15#00",
            # try switching to invalid pid
            "read packet: $Hgp{:x}.{:x}#00".format(pid+1, tid),
            "send packet: $Eff#00",
            "read packet: $Hcp{:x}.{:x}#00".format(pid+1, tid),
            "send packet: $Eff#00",
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_detach_current(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets(["multiprocess+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("multiprocess+", ret["qSupported_response"])
        self.reset_test_sequence()

        # get process pid
        self.test_sequence.add_log_lines([
            "read packet: $qProcessInfo#00",
            {"direction": "send", "regex": self.procinfo_regex,
             "capture": {1: "pid"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        pid = ret["pid"]
        self.reset_test_sequence()

        # detach the process
        self.test_sequence.add_log_lines([
            "read packet: $D;{}#00".format(pid),
            "send packet: $OK#00",
            "read packet: $qC#00",
            "send packet: $E44#00",
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_detach_all(self):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["fork"])
        self.add_qSupported_packets(["multiprocess+",
                                     "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format("fork"),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        parent_tid = ret["parent_tid"]
        child_pid = ret["child_pid"]
        child_tid = ret["child_tid"]
        self.reset_test_sequence()

        self.test_sequence.add_log_lines([
            # double-check our PIDs
            "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: $OK#00",
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: $OK#00",
            # detach all processes
            "read packet: $D#00",
            "send packet: $OK#00",
            # verify that both PIDs are invalid now
            "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: $Eff#00",
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: $Eff#00",
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_kill_all(self):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["fork"])
        self.add_qSupported_packets(["multiprocess+",
                                     "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format("fork"),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        child_pid = ret["child_pid"]
        self.reset_test_sequence()

        exit_regex = "[$]X09;process:([0-9a-f]+)#.*"
        self.test_sequence.add_log_lines([
            # kill all processes
            "read packet: $k#00",
            {"direction": "send", "regex": exit_regex,
             "capture": {1: "pid1"}},
            {"direction": "send", "regex": exit_regex,
             "capture": {1: "pid2"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        self.assertEqual(set([ret["pid1"], ret["pid2"]]),
                         set([parent_pid, child_pid]))

    def vkill_test(self, kill_parent=False, kill_child=False):
        assert kill_parent or kill_child
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["fork"])
        self.add_qSupported_packets(["multiprocess+",
                                     "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format("fork"),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        parent_tid = ret["parent_tid"]
        child_pid = ret["child_pid"]
        child_tid = ret["child_tid"]
        self.reset_test_sequence()

        if kill_parent:
            self.test_sequence.add_log_lines([
                # kill the process
                "read packet: $vKill;{}#00".format(parent_pid),
                "send packet: $OK#00",
            ], True)
        if kill_child:
            self.test_sequence.add_log_lines([
                # kill the process
                "read packet: $vKill;{}#00".format(child_pid),
                "send packet: $OK#00",
            ], True)
        self.test_sequence.add_log_lines([
            # check child PID/TID
            "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
            "send packet: ${}#00".format("Eff" if kill_child else "OK"),
            # check parent PID/TID
            "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
            "send packet: ${}#00".format("Eff" if kill_parent else "OK"),
        ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vkill_child(self):
        self.vkill_test(kill_child=True)

    @add_test_categories(["fork"])
    def test_vkill_parent(self):
        self.vkill_test(kill_parent=True)

    @add_test_categories(["fork"])
    def test_vkill_both(self):
        self.vkill_test(kill_parent=True, kill_child=True)

    def resume_one_test(self, run_order):
        self.build()
        self.prep_debug_monitor_and_inferior(inferior_args=["fork", "trap"])
        self.add_qSupported_packets(["multiprocess+",
                                     "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines([
            "read packet: $c#00",
            {"direction": "send", "regex": self.fork_regex.format("fork"),
             "capture": self.fork_capture},
        ], True)
        ret = self.expect_gdbremote_sequence()
        parent_pid = ret["parent_pid"]
        parent_tid = ret["parent_tid"]
        child_pid = ret["child_pid"]
        child_tid = ret["child_tid"]
        self.reset_test_sequence()

        parent_expect = [
            "[$]T05thread:p{}.{};.*".format(parent_pid, parent_tid),
            "[$]W00;process:{}#.*".format(parent_pid),
        ]
        child_expect = [
            "[$]T05thread:p{}.{};.*".format(child_pid, child_tid),
            "[$]W00;process:{}#.*".format(child_pid),
        ]

        for x in run_order:
            if x == "parent":
                pidtid = (parent_pid, parent_tid)
                expect = parent_expect.pop(0)
            elif x == "child":
                pidtid = (child_pid, child_tid)
                expect = child_expect.pop(0)
            else:
                assert False, "unexpected x={}".format(x)

            self.test_sequence.add_log_lines([
                # continue the selected process
                "read packet: $Hcp{}.{}#00".format(*pidtid),
                "send packet: $OK#00",
                "read packet: $c#00",
                {"direction": "send", "regex": expect},
            ], True)
            # if at least one process remained, check both PIDs
            if parent_expect or child_expect:
                self.test_sequence.add_log_lines([
                    "read packet: $Hgp{}.{}#00".format(parent_pid, parent_tid),
                    "send packet: ${}#00".format("OK" if parent_expect else "Eff"),
                    "read packet: $Hgp{}.{}#00".format(child_pid, child_tid),
                    "send packet: ${}#00".format("OK" if child_expect else "Eff"),
                ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_c_parent(self):
        self.resume_one_test(run_order=["parent", "parent"])

    @add_test_categories(["fork"])
    def test_c_child(self):
        self.resume_one_test(run_order=["child", "child"])

    @add_test_categories(["fork"])
    def test_c_parent_then_child(self):
        self.resume_one_test(run_order=["parent", "parent", "child", "child"])

    @add_test_categories(["fork"])
    def test_c_child_then_parent(self):
        self.resume_one_test(run_order=["child", "child", "parent", "parent"])

    @add_test_categories(["fork"])
    def test_c_interspersed(self):
        self.resume_one_test(run_order=["parent", "child", "parent", "child"])
