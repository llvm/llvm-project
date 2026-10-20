import json
import re

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
@skipIfMTE  # MTE security transition shims restrict socket operations.
class TestGdbRemoteThreadsInStopReply(gdbremote_testcase.GdbRemoteTestCaseBase):
    ENABLE_THREADS_IN_STOP_REPLY_ENTRIES = [
        "read packet: $QListThreadsInStopReply#21",
        "send packet: $OK#00",
    ]

    def gather_stop_reply_fields(self, thread_count, field_names):
        context, threads = self.launch_with_threads(thread_count)
        key_vals_text = context.get("stop_reply_kv")
        self.assertIsNotNone(key_vals_text)

        self.reset_test_sequence()
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        hw_info = self.parse_hw_info(context)

        # Parse the stop reply contents.
        kv_dict = self.parse_key_val_dict(key_vals_text)

        result = dict()
        result["pc_register"] = hw_info["pc_register"]
        result["little_endian"] = hw_info["little_endian"]
        result["ptrsize"] = hw_info["ptrsize"]
        for key_field in field_names:
            result[key_field] = kv_dict.get(key_field)

        return result

    def gather_stop_reply_threads(self, thread_count):
        # Pull out threads from stop response.
        stop_reply_threads_text = self.gather_stop_reply_fields(
            thread_count, ["threads"]
        )["threads"]
        if stop_reply_threads_text:
            return [
                int(thread_id, 16) for thread_id in stop_reply_threads_text.split(",")
            ]
        else:
            return []

    def gather_stop_reply_pcs(self, thread_count):
        results = self.gather_stop_reply_fields(thread_count, ["threads", "thread-pcs"])
        if not results:
            return []

        threads_text = results["threads"]
        pcs_text = results["thread-pcs"]
        thread_ids = threads_text.split(",")
        pcs = pcs_text.split(",")
        self.assertEqual(len(thread_ids), len(pcs))

        thread_pcs = dict()
        for i in range(0, len(pcs)):
            thread_pcs[int(thread_ids[i], 16)] = pcs[i]

        result = dict()
        result["thread_pcs"] = thread_pcs
        result["pc_register"] = results["pc_register"]
        result["little_endian"] = results["little_endian"]
        return result

    def switch_endian(self, egg):
        return "".join(reversed(re.findall("..", egg)))

    def parse_hw_info(self, context):
        self.assertIsNotNone(context)
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        reg_info = self.parse_register_info_packets(context)
        (pc_lldb_reg_index, pc_reg_info) = self.find_pc_reg_info(reg_info)

        hw_info = dict()
        hw_info["pc_register"] = pc_lldb_reg_index
        hw_info["little_endian"] = endian == "little"
        hw_info["ptrsize"] = int(process_info.get("ptrsize", 0))
        return hw_info

    def gather_threads_info(self):
        """The parsed jThreadsInfo reply: one dict per thread."""
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $jThreadsInfo#c1",
                {
                    "direction": "send",
                    "regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                    "capture": {1: "threads_info"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        threads_info = context.get("threads_info")
        # A literal '}' is escaped as '}]' on the wire.
        return json.loads(re.sub(r"}]", "}", threads_info))

    def gather_threads_info_pcs(self, pc_register, little_endian):
        jthreads_info = self.gather_threads_info()
        register = str(pc_register)
        thread_pcs = dict()
        for thread_info in jthreads_info:
            tid = thread_info["tid"]
            pc = thread_info["registers"][register]
            thread_pcs[tid] = self.switch_endian(pc) if little_endian else pc

        return thread_pcs

    def test_QListThreadsInStopReply_supported(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, True
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipIfNetBSD
    @expectedFailureAll(oslist=["windows"])  # Extra threads present
    def test_stop_reply_reports_multiple_threads(self):
        self.build()
        self.set_inferior_startup_launch()
        # Gather threads from stop notification when QThreadsInStopReply is
        # enabled.
        self.test_sequence.add_log_lines(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, True
        )
        stop_reply_threads = self.gather_stop_reply_threads(5)
        self.assertEqual(len(stop_reply_threads), 5)

    @skipIfNetBSD
    def test_no_QListThreadsInStopReply_supplies_no_threads(self):
        self.build()
        self.set_inferior_startup_launch()
        # Gather threads from stop notification when QThreadsInStopReply is not
        # enabled.
        stop_reply_threads = self.gather_stop_reply_threads(5)
        self.assertEqual(len(stop_reply_threads), 0)

    @skipIfNetBSD
    def test_stop_reply_reports_correct_threads(self):
        self.build()
        self.set_inferior_startup_launch()
        # Gather threads from stop notification when QThreadsInStopReply is
        # enabled.
        thread_count = 5
        self.test_sequence.add_log_lines(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, True
        )
        stop_reply_threads = self.gather_stop_reply_threads(thread_count)

        # Gather threads from q{f,s}ThreadInfo.
        self.reset_test_sequence()
        self.add_threadinfo_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)
        self.assertGreaterEqual(len(threads), thread_count)

        # Ensure each thread in q{f,s}ThreadInfo appears in stop reply threads
        for tid in threads:
            self.assertIn(tid, stop_reply_threads)

    @add_test_categories(["debugserver"])
    def test_stop_reply_expedites_frame_pointer_backchain(self):
        """The stop reply expedites at most the two innermost backchain
        entries, one `memory:` entry each of 2 * ptrsize bytes."""
        self.build()
        self.set_inferior_startup_launch()
        results = self.gather_stop_reply_fields(1, ["memory"])

        memory = results["memory"]
        self.assertIsNotNone(memory, "the stop reply carried no memory: entry")
        # parse_key_val_dict promotes a repeated key to a list.
        entries = memory if isinstance(memory, list) else [memory]

        # A backchain entry is 2 pointers at $fp: the previous FP and PC.
        backchain_entry_size = 2 * results["ptrsize"]
        for entry in entries:
            addr, sep, hex_bytes = entry.partition("=")
            self.assertEqual(sep, "=", "malformed memory entry %r" % entry)
            self.assertEqual(
                len(hex_bytes),
                2 * backchain_entry_size,
                "memory:%s= should carry %d bytes (2 * ptrsize), got %d: %r"
                % (addr, backchain_entry_size, len(hex_bytes) // 2, entry),
            )
        # The stop reply caps the backchain at 2 entries.  How many it actually
        # walks depends on where the backchain terminates.
        self.assertLessEqual(len(entries), 2, "expected at most 2 backchain entries")

    # Frame 0's stack memory is expedited only by a debugserver built from this
    # tree; an older system debugserver sends the backchain alone.
    @skipIfOutOfTreeDebugserver
    @add_test_categories(["debugserver"])
    def test_threads_info_expedites_stopped_frame_stack(self):
        """jThreadsInfo expedites every thread's frame pointer backchain, plus
        frame 0's stack memory for the stopped thread only."""
        self.build()
        self.set_inferior_startup_launch()
        results = self.gather_stop_reply_fields(5, ["thread"])
        backchain_entry_size = 2 * results["ptrsize"]
        stopped_tid = int(results["thread"], 16)

        # Chunk sizes tell the two apart: a backchain entry is 2 pointers,
        # frame 0's stack memory is a larger range.
        def classify(thread_info):
            sizes = [len(m["bytes"]) // 2 for m in thread_info.get("memory", [])]
            stack_backchain = [n for n in sizes if n == backchain_entry_size]
            frame_0_stack_memory = [n for n in sizes if n != backchain_entry_size]
            return sizes, stack_backchain, frame_0_stack_memory

        saw_stopped = False
        for thread_info in self.gather_threads_info():
            sizes, stack_backchain, frame_0_stack_memory = classify(thread_info)
            where = "thread %x, chunk sizes %s" % (thread_info["tid"], sizes)
            self.assertGreaterEqual(
                len(stack_backchain), 1, "no backchain entry for %s" % where
            )
            if thread_info["tid"] == stopped_tid:
                saw_stopped = True
                # One chunk, or two when $fp is usable.
                self.assertIn(
                    len(frame_0_stack_memory),
                    (1, 2),
                    "no frame 0 stack memory for stopped %s" % where,
                )
            else:
                self.assertEqual(
                    frame_0_stack_memory,
                    [],
                    "unexpected frame 0 stack memory for %s" % where,
                )
        self.assertTrue(saw_stopped, "no jThreadsInfo entry for the stopped thread")

    @skipIfNetBSD
    @skipIfWindows  # Flaky on Windows
    def test_stop_reply_contains_thread_pcs(self):
        self.build()
        self.set_inferior_startup_launch()
        thread_count = 5
        self.test_sequence.add_log_lines(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, True
        )
        results = self.gather_stop_reply_pcs(thread_count)
        stop_reply_pcs = results["thread_pcs"]
        pc_register = results["pc_register"]
        little_endian = results["little_endian"]
        self.assertGreaterEqual(len(stop_reply_pcs), thread_count)

        threads_info_pcs = self.gather_threads_info_pcs(pc_register, little_endian)

        self.assertEqual(len(threads_info_pcs), len(stop_reply_pcs))
        for thread_id in stop_reply_pcs:
            self.assertIn(thread_id, threads_info_pcs)
            self.assertEqual(
                int(stop_reply_pcs[thread_id], 16), int(threads_info_pcs[thread_id], 16)
            )
