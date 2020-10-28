
import json
import re

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteThreadsInStopReply(
        gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    ENABLE_THREADS_IN_STOP_REPLY_ENTRIES = [
        "read packet: $QListThreadsInStopReply#21",
        "send packet: $OK#00",
    ]

    def gather_stop_reply_fields(self, post_startup_log_lines, thread_count,
            field_names):
        # Set up the inferior args.
        inferior_args = []
        for i in range(thread_count - 1):
            inferior_args.append("thread:new")
        inferior_args.append("sleep:10")
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=inferior_args)

        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        # Assumes test_sequence has anything added needed to setup the initial state.
        # (Like optionally enabling QThreadsInStopReply.)
        if post_startup_log_lines:
            self.test_sequence.add_log_lines(post_startup_log_lines, True)
        self.test_sequence.add_log_lines([
            "read packet: $c#63"
        ], True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        hw_info = self.parse_hw_info(context)

        # Give threads time to start up, then break.
        time.sleep(self.DEFAULT_SLEEP)
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: {}".format(
                    chr(3)),
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$",
                    "capture": {
                        1: "stop_result",
                        2: "key_vals_text"}},
            ],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Wait until all threads have started.
        threads = self.wait_for_thread_count(thread_count)
        self.assertIsNotNone(threads)
        self.assertEqual(len(threads), thread_count)

        # Run, then stop the process, grab the stop reply content.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(["read packet: $c#63",
                                          "read packet: {}".format(chr(3)),
                                          {"direction": "send",
                                           "regex": r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$",
                                           "capture": {1: "stop_result",
                                                       2: "key_vals_text"}},
                                          ],
                                         True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Parse the stop reply contents.
        key_vals_text = context.get("key_vals_text")
        self.assertIsNotNone(key_vals_text)
        kv_dict = self.parse_key_val_dict(key_vals_text)
        self.assertIsNotNone(kv_dict)

        result = dict();
        result["pc_register"] = hw_info["pc_register"]
        result["little_endian"] = hw_info["little_endian"]
        for key_field in field_names:
            result[key_field] = kv_dict.get(key_field)

        return result

    def gather_stop_reply_threads(self, post_startup_log_lines, thread_count):
        # Pull out threads from stop response.
        stop_reply_threads_text = self.gather_stop_reply_fields(
                post_startup_log_lines, thread_count, ["threads"])["threads"]
        if stop_reply_threads_text:
            return [int(thread_id, 16)
                    for thread_id in stop_reply_threads_text.split(",")]
        else:
            return []

    def gather_stop_reply_pcs(self, post_startup_log_lines, thread_count):
        results = self.gather_stop_reply_fields( post_startup_log_lines,
                thread_count, ["threads", "thread-pcs"])
        if not results:
            return []

        threads_text = results["threads"]
        pcs_text = results["thread-pcs"]
        thread_ids = threads_text.split(",")
        pcs = pcs_text.split(",")
        self.assertEquals(len(thread_ids), len(pcs))

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
        hw_info["little_endian"] = (endian == "little")
        return hw_info

    def gather_threads_info_pcs(self, pc_register, little_endian):
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
                [
                    "read packet: $jThreadsInfo#c1",
                    {
                        "direction": "send",
                        "regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                        "capture": {
                            1: "threads_info"}},
                ],
                True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        threads_info = context.get("threads_info")
        register = str(pc_register)
        # The jThreadsInfo response is not valid JSON data, so we have to
        # clean it up first.
        jthreads_info = json.loads(re.sub(r"}]", "}", threads_info))
        thread_pcs = dict()
        for thread_info in jthreads_info:
            tid = thread_info["tid"]
            pc = thread_info["registers"][register]
            thread_pcs[tid] = self.switch_endian(pc) if little_endian else pc

        return thread_pcs

    def QListThreadsInStopReply_supported(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_QListThreadsInStopReply_supported_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.QListThreadsInStopReply_supported()

    @llgs_test
    def test_QListThreadsInStopReply_supported_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.QListThreadsInStopReply_supported()

    def stop_reply_reports_multiple_threads(self, thread_count):
        # Gather threads from stop notification when QThreadsInStopReply is
        # enabled.
        stop_reply_threads = self.gather_stop_reply_threads(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, thread_count)
        self.assertEqual(len(stop_reply_threads), thread_count)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_stop_reply_reports_multiple_threads_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_reports_multiple_threads(5)

    # In current implementation of llgs on Windows, as a response to '\x03' packet, the debugger
    # of the native process will trigger a call to DebugBreakProcess that will create a new thread
    # to handle the exception debug event. So one more stop thread will be notified to the
    # delegate, e.g. llgs.  So tests below to assert the stop threads number will all fail.
    @expectedFailureAll(oslist=["windows"])
    @skipIfNetBSD
    @llgs_test
    def test_stop_reply_reports_multiple_threads_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_reports_multiple_threads(5)

    def no_QListThreadsInStopReply_supplies_no_threads(self, thread_count):
        # Gather threads from stop notification when QThreadsInStopReply is not
        # enabled.
        stop_reply_threads = self.gather_stop_reply_threads(None, thread_count)
        self.assertEqual(len(stop_reply_threads), 0)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_no_QListThreadsInStopReply_supplies_no_threads_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.no_QListThreadsInStopReply_supplies_no_threads(5)

    @expectedFailureAll(oslist=["windows"])
    @skipIfNetBSD
    @llgs_test
    def test_no_QListThreadsInStopReply_supplies_no_threads_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.no_QListThreadsInStopReply_supplies_no_threads(5)

    def stop_reply_reports_correct_threads(self, thread_count):
        # Gather threads from stop notification when QThreadsInStopReply is
        # enabled.
        stop_reply_threads = self.gather_stop_reply_threads(
            self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, thread_count)
        self.assertEqual(len(stop_reply_threads), thread_count)

        # Gather threads from q{f,s}ThreadInfo.
        self.reset_test_sequence()
        self.add_threadinfo_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)
        self.assertEqual(len(threads), thread_count)

        # Ensure each thread in q{f,s}ThreadInfo appears in stop reply threads
        for tid in threads:
            self.assertTrue(tid in stop_reply_threads)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_stop_reply_reports_correct_threads_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_reports_correct_threads(5)

    @expectedFailureAll(oslist=["windows"])
    @skipIfNetBSD
    @llgs_test
    def test_stop_reply_reports_correct_threads_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_reports_correct_threads(5)

    def stop_reply_contains_thread_pcs(self, thread_count):
        results = self.gather_stop_reply_pcs(
                self.ENABLE_THREADS_IN_STOP_REPLY_ENTRIES, thread_count)
        stop_reply_pcs = results["thread_pcs"]
        pc_register = results["pc_register"]
        little_endian = results["little_endian"]
        self.assertEqual(len(stop_reply_pcs), thread_count)

        threads_info_pcs = self.gather_threads_info_pcs(pc_register,
                little_endian)

        self.assertEqual(len(threads_info_pcs), thread_count)
        for thread_id in stop_reply_pcs:
            self.assertTrue(thread_id in threads_info_pcs)
            self.assertTrue(int(stop_reply_pcs[thread_id], 16)
                    == int(threads_info_pcs[thread_id], 16))

    @expectedFailureAll(oslist=["windows"])
    @skipIfNetBSD
    @llgs_test
    def test_stop_reply_contains_thread_pcs_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_contains_thread_pcs(5)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @debugserver_test
    def test_stop_reply_contains_thread_pcs_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_reply_contains_thread_pcs(5)
