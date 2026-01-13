import random

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.support import seven

from fork_testbase import GdbRemoteForkTestBase


class TestGdbRemoteFork(GdbRemoteForkTestBase):
    def setUp(self):
        GdbRemoteForkTestBase.setUp(self)
        if self.getPlatform() == "linux" and self.getArchitecture() in [
            "arm",
            "aarch64",
        ]:
            self.skipTest("Unsupported for Arm/AArch64 Linux")

    @add_test_categories(["fork"])
    def test_fork_multithreaded(self):
        _, _, child_pid, _ = self.start_fork_test(["thread:new"] * 2 + ["fork"])

        # detach the forked child
        self.test_sequence.add_log_lines(
            [
                "read packet: $D;{}#00".format(child_pid),
                "send packet: $OK#00",
                "read packet: $k#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_fork(self):
        parent_pid, _ = self.fork_and_detach_test("fork")

        # resume the parent
        self.test_sequence.add_log_lines(
            [
                "read packet: $c#00",
                "send packet: $W00;process:{}#00".format(parent_pid),
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vfork(self):
        parent_pid, parent_tid = self.fork_and_detach_test("vfork")

        # resume the parent
        self.test_sequence.add_log_lines(
            [
                "read packet: $c#00",
                {
                    "direction": "send",
                    "regex": r"[$]T[0-9a-fA-F]{{2}}thread:p{}[.]{}.*vforkdone.*".format(
                        parent_pid, parent_tid
                    ),
                },
                "read packet: $c#00",
                "send packet: $W00;process:{}#00".format(parent_pid),
            ],
            True,
        )
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
        self.test_sequence.add_log_lines(
            [
                "read packet: $qC#00",
                {
                    "direction": "send",
                    "regex": "[$]QCp([0-9a-f]+).([0-9a-f]+)#.*",
                    "capture": {1: "pid", 2: "tid"},
                },
            ],
            True,
        )
        ret = self.expect_gdbremote_sequence()
        pid, tid = (int(ret[x], 16) for x in ("pid", "tid"))
        self.reset_test_sequence()

        self.test_sequence.add_log_lines(
            [
                # try switching to correct pid
                "read packet: $Hgp{:x}.{:x}#00".format(pid, tid),
                "send packet: $OK#00",
                "read packet: $Hcp{:x}.{:x}#00".format(pid, tid),
                "send packet: $OK#00",
                # try switching to invalid tid
                "read packet: $Hgp{:x}.{:x}#00".format(pid, tid + 1),
                "send packet: $E15#00",
                "read packet: $Hcp{:x}.{:x}#00".format(pid, tid + 1),
                "send packet: $E15#00",
                # try switching to invalid pid
                "read packet: $Hgp{:x}.{:x}#00".format(pid + 1, tid),
                "send packet: $Eff#00",
                "read packet: $Hcp{:x}.{:x}#00".format(pid + 1, tid),
                "send packet: $Eff#00",
            ],
            True,
        )
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
        self.test_sequence.add_log_lines(
            [
                "read packet: $qC#00",
                {
                    "direction": "send",
                    "regex": "[$]QCp([0-9a-f]+).[0-9a-f]+#.*",
                    "capture": {1: "pid"},
                },
            ],
            True,
        )
        ret = self.expect_gdbremote_sequence()
        pid = ret["pid"]
        self.reset_test_sequence()

        # detach the process
        self.test_sequence.add_log_lines(
            [
                "read packet: $D;{}#00".format(pid),
                "send packet: $OK#00",
                "read packet: $qC#00",
                "send packet: $E44#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_detach_all(self):
        self.detach_all_test()

    @add_test_categories(["fork"])
    def test_kill_all(self):
        parent_pid, _, child_pid, _ = self.start_fork_test(["fork"])

        exit_regex = "[$]X09;process:([0-9a-f]+)#.*"
        self.test_sequence.add_log_lines(
            [
                # kill all processes
                "read packet: $k#00",
                {"direction": "send", "regex": exit_regex, "capture": {1: "pid1"}},
                {"direction": "send", "regex": exit_regex, "capture": {1: "pid2"}},
            ],
            True,
        )
        ret = self.expect_gdbremote_sequence()
        self.assertEqual(set([ret["pid1"], ret["pid2"]]), set([parent_pid, child_pid]))

    @add_test_categories(["fork"])
    def test_vkill_child(self):
        self.vkill_test(kill_child=True)

    @add_test_categories(["fork"])
    def test_vkill_parent(self):
        self.vkill_test(kill_parent=True)

    @add_test_categories(["fork"])
    def test_vkill_both(self):
        self.vkill_test(kill_parent=True, kill_child=True)

    @add_test_categories(["fork"])
    def test_vCont_two_processes(self):
        parent_pid, parent_tid, child_pid, child_tid = self.start_fork_test(
            ["fork", "stop"]
        )

        self.test_sequence.add_log_lines(
            [
                # try to resume both processes
                "read packet: $vCont;c:p{}.{};c:p{}.{}#00".format(
                    parent_pid, parent_tid, child_pid, child_tid
                ),
                "send packet: $E03#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vCont_all_processes_explicit(self):
        self.start_fork_test(["fork", "stop"])

        self.test_sequence.add_log_lines(
            [
                # try to resume all processes implicitly
                "read packet: $vCont;c:p-1.-1#00",
                "send packet: $E03#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_vCont_all_processes_implicit(self):
        self.start_fork_test(["fork", "stop"])

        self.test_sequence.add_log_lines(
            [
                # try to resume all processes implicitly
                "read packet: $vCont;c#00",
                "send packet: $E03#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_threadinfo(self):
        parent_pid, parent_tid, child_pid, child_tid = self.start_fork_test(
            ["fork", "thread:new", "stop"]
        )
        pidtids = [
            (parent_pid, parent_tid),
            (child_pid, child_tid),
        ]

        self.add_threadinfo_collection_packets()
        ret = self.expect_gdbremote_sequence()
        prev_pidtids = set(self.parse_threadinfo_packets(ret))
        self.assertEqual(
            prev_pidtids,
            frozenset((int(pid, 16), int(tid, 16)) for pid, tid in pidtids),
        )
        self.reset_test_sequence()

        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hcp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $c#00",
                    {
                        "direction": "send",
                        "regex": self.stop_regex.format(*pidtid),
                    },
                ],
                True,
            )
            self.add_threadinfo_collection_packets()
            ret = self.expect_gdbremote_sequence()
            self.reset_test_sequence()
            new_pidtids = set(self.parse_threadinfo_packets(ret))
            added_pidtid = new_pidtids - prev_pidtids
            prev_pidtids = new_pidtids

            # verify that we've got exactly one new thread, and that
            # the PID matches
            self.assertEqual(len(added_pidtid), 1)
            self.assertEqual(added_pidtid.pop()[0], int(pidtid[0], 16))

        for pidtid in new_pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                ],
                True,
            )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_memory_read_write(self):
        self.build()
        INITIAL_DATA = "Initial message"
        self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "set-message:{}".format(INITIAL_DATA),
                "get-data-address-hex:g_message",
                "fork",
                "print-message:",
                "stop",
            ]
        )
        self.add_qSupported_packets(["multiprocess+", "fork-events+"])
        ret = self.expect_gdbremote_sequence()
        self.assertIn("fork-events+", ret["qSupported_response"])
        self.reset_test_sequence()

        # continue and expect fork
        self.test_sequence.add_log_lines(
            [
                "read packet: $c#00",
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"data address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "addr"},
                },
                {
                    "direction": "send",
                    "regex": self.fork_regex.format("fork"),
                    "capture": self.fork_capture,
                },
            ],
            True,
        )
        ret = self.expect_gdbremote_sequence()
        pidtids = {
            "parent": (ret["parent_pid"], ret["parent_tid"]),
            "child": (ret["child_pid"], ret["child_tid"]),
        }
        addr = ret["addr"]
        self.reset_test_sequence()

        for name, pidtid in pidtids.items():
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    # read the current memory contents
                    "read packet: $m{},{:x}#00".format(addr, len(INITIAL_DATA) + 1),
                    {
                        "direction": "send",
                        "regex": r"^[$](.+)#.*$",
                        "capture": {1: "data"},
                    },
                    # write a new value
                    "read packet: $M{},{:x}:{}#00".format(
                        addr, len(name) + 1, seven.hexlify(name + "\0")
                    ),
                    "send packet: $OK#00",
                    # resume the process and wait for the trap
                    "read packet: $Hcp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $c#00",
                    {
                        "type": "output_match",
                        "regex": self.maybe_strict_output_regex(r"message: (.*)\r\n"),
                        "capture": {1: "printed_message"},
                    },
                    {
                        "direction": "send",
                        "regex": self.stop_regex.format(*pidtid),
                    },
                ],
                True,
            )
            ret = self.expect_gdbremote_sequence()
            data = seven.unhexlify(ret["data"])
            self.assertEqual(data, INITIAL_DATA + "\0")
            self.assertEqual(ret["printed_message"], name)
            self.reset_test_sequence()

        # we do the second round separately to make sure that initial data
        # is correctly preserved while writing into the first process

        for name, pidtid in pidtids.items():
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    # read the current memory contents
                    "read packet: $m{},{:x}#00".format(addr, len(name) + 1),
                    {
                        "direction": "send",
                        "regex": r"^[$](.+)#.*$",
                        "capture": {1: "data"},
                    },
                ],
                True,
            )
            ret = self.expect_gdbremote_sequence()
            self.assertIsNotNone(ret.get("data"))
            data = seven.unhexlify(ret.get("data"))
            self.assertEqual(data, name + "\0")
            self.reset_test_sequence()

    @add_test_categories(["fork"])
    def test_register_read_write(self):
        parent_pid, parent_tid, child_pid, child_tid = self.start_fork_test(
            ["fork", "thread:new", "stop"]
        )
        pidtids = [
            (parent_pid, parent_tid),
            (child_pid, child_tid),
        ]

        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hcp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $c#00",
                    {
                        "direction": "send",
                        "regex": self.stop_regex.format(*pidtid),
                    },
                ],
                True,
            )

        self.add_threadinfo_collection_packets()
        ret = self.expect_gdbremote_sequence()
        self.reset_test_sequence()

        pidtids = set(self.parse_threadinfo_packets(ret))
        self.assertEqual(len(pidtids), 4)
        # first, save register values from all the threads
        thread_regs = {}
        for pidtid in pidtids:
            for regno in range(256):
                self.test_sequence.add_log_lines(
                    [
                        "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                        "send packet: $OK#00",
                        "read packet: $p{:x}#00".format(regno),
                        {
                            "direction": "send",
                            "regex": r"^[$](.+)#.*$",
                            "capture": {1: "data"},
                        },
                    ],
                    True,
                )
                ret = self.expect_gdbremote_sequence()
                data = ret.get("data")
                self.assertIsNotNone(data)
                # ignore registers shorter than 32 bits (this also catches
                # "Exx" errors)
                if len(data) >= 8:
                    break
            else:
                self.skipTest("no usable register found")
            thread_regs[pidtid] = (regno, data)

        vals = set(x[1] for x in thread_regs.values())
        # NB: cheap hack to make the loop below easier
        new_val = next(iter(vals))

        # then, start altering them and verify that we don't unexpectedly
        # change the value from another thread
        for pidtid in pidtids:
            old_val = thread_regs[pidtid]
            regno = old_val[0]
            old_val_length = len(old_val[1])
            # generate a unique new_val
            while new_val in vals:
                new_val = "{{:0{}x}}".format(old_val_length).format(
                    random.getrandbits(old_val_length * 4)
                )
            vals.add(new_val)

            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $p{:x}#00".format(regno),
                    {
                        "direction": "send",
                        "regex": r"^[$](.+)#.*$",
                        "capture": {1: "data"},
                    },
                    "read packet: $P{:x}={}#00".format(regno, new_val),
                    "send packet: $OK#00",
                ],
                True,
            )
            ret = self.expect_gdbremote_sequence()
            data = ret.get("data")
            self.assertIsNotNone(data)
            self.assertEqual(data, old_val[1])
            thread_regs[pidtid] = (regno, new_val)

        # finally, verify that new values took effect
        for pidtid in pidtids:
            old_val = thread_regs[pidtid]
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $p{:x}#00".format(old_val[0]),
                    {
                        "direction": "send",
                        "regex": r"^[$](.+)#.*$",
                        "capture": {1: "data"},
                    },
                ],
                True,
            )
            ret = self.expect_gdbremote_sequence()
            data = ret.get("data")
            self.assertIsNotNone(data)
            self.assertEqual(data, old_val[1])

    @add_test_categories(["fork"])
    def test_qC(self):
        parent_pid, parent_tid, child_pid, child_tid = self.start_fork_test(
            ["fork", "thread:new", "stop"]
        )
        pidtids = [
            (parent_pid, parent_tid),
            (child_pid, child_tid),
        ]

        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hcp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $c#00",
                    {
                        "direction": "send",
                        "regex": self.stop_regex.format(*pidtid),
                    },
                ],
                True,
            )

        self.add_threadinfo_collection_packets()
        ret = self.expect_gdbremote_sequence()
        self.reset_test_sequence()

        pidtids = set(self.parse_threadinfo_packets(ret))
        self.assertEqual(len(pidtids), 4)
        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $qC#00",
                    "send packet: $QCp{:x}.{:x}#00".format(*pidtid),
                ],
                True,
            )
        self.expect_gdbremote_sequence()

    @add_test_categories(["fork"])
    def test_T(self):
        parent_pid, parent_tid, child_pid, child_tid = self.start_fork_test(
            ["fork", "thread:new", "stop"]
        )
        pidtids = [
            (parent_pid, parent_tid),
            (child_pid, child_tid),
        ]

        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hcp{}.{}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $c#00",
                    {
                        "direction": "send",
                        "regex": self.stop_regex.format(*pidtid),
                    },
                ],
                True,
            )

        self.add_threadinfo_collection_packets()
        ret = self.expect_gdbremote_sequence()
        self.reset_test_sequence()

        pidtids = set(self.parse_threadinfo_packets(ret))
        self.assertEqual(len(pidtids), 4)
        max_pid = max(pid for pid, tid in pidtids)
        max_tid = max(tid for pid, tid in pidtids)
        bad_pidtids = (
            (max_pid, max_tid + 1, "E02"),
            (max_pid + 1, max_tid, "E01"),
            (max_pid + 1, max_tid + 1, "E01"),
        )

        for pidtid in pidtids:
            self.test_sequence.add_log_lines(
                [
                    # test explicit PID+TID
                    "read packet: $Tp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                    # test implicit PID via Hg
                    "read packet: $Hgp{:x}.{:x}#00".format(*pidtid),
                    "send packet: $OK#00",
                    "read packet: $T{:x}#00".format(max_tid + 1),
                    "send packet: $E02#00",
                    "read packet: $T{:x}#00".format(pidtid[1]),
                    "send packet: $OK#00",
                ],
                True,
            )
        for pid, tid, expected in bad_pidtids:
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Tp{:x}.{:x}#00".format(pid, tid),
                    "send packet: ${}#00".format(expected),
                ],
                True,
            )
        self.expect_gdbremote_sequence()
