import unittest2
import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemote_qThreadStopInfo(gdbremote_testcase.GdbRemoteTestCaseBase):
    THREAD_COUNT = 5

    def gather_stop_replies_via_qThreadStopInfo(self, threads):
        # Grab stop reply for each thread via qThreadStopInfo{tid:hex}.
        stop_replies = {}
        thread_dicts = {}
        for thread in threads:
            # Run the qThreadStopInfo command.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $qThreadStopInfo{:x}#00".format(thread),
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

            # Parse stop reply contents.
            key_vals_text = context.get("key_vals_text")
            self.assertIsNotNone(key_vals_text)
            kv_dict = self.parse_key_val_dict(key_vals_text)
            self.assertIsNotNone(kv_dict)

            # Verify there is a thread and that it matches the expected thread
            # id.
            kv_thread = kv_dict.get("thread")
            self.assertIsNotNone(kv_thread)
            kv_thread_id = int(kv_thread, 16)
            self.assertEqual(kv_thread_id, thread)

            # Grab the stop id reported.
            stop_result_text = context.get("stop_result")
            self.assertIsNotNone(stop_result_text)
            stop_replies[kv_thread_id] = int(stop_result_text, 16)

            # Hang on to the key-val dictionary for the thread.
            thread_dicts[kv_thread_id] = kv_dict

        return stop_replies

    @skipIfNetBSD
    def test_qThreadStopInfo_works_for_multiple_threads(self):
        self.build()
        self.set_inferior_startup_launch()
        _, threads = self.launch_with_threads(self.THREAD_COUNT)
        stop_replies = self.gather_stop_replies_via_qThreadStopInfo(threads)
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        # Consider one more thread created by calling DebugBreakProcess.
        if re.match(".*-.*-windows", triple):
            self.assertGreaterEqual(len(stop_replies), self.THREAD_COUNT)
        else:
            self.assertEqual(len(stop_replies), self.THREAD_COUNT)

    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48418")
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["windows"]) # Output forwarding not implemented
    def test_qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new"]*4 + ["stop-me-now", "sleep:60"])

        self.test_sequence.add_log_lines([
                "read packet: $c#00",
                {"type": "output_match",
                    "regex": self.maybe_strict_output_regex(r"stop-me-now\r\n")},
                "read packet: \x03",
                {"direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})([^#]*)#..$"}], True)
        self.add_threadinfo_collection_packets()
        context = self.expect_gdbremote_sequence()
        threads = self.parse_threadinfo_packets(context)

        stop_replies = self.gather_stop_replies_via_qThreadStopInfo(threads)
        self.assertIsNotNone(stop_replies)

        no_stop_reason_count = sum(
            1 for stop_reason in list(
                stop_replies.values()) if stop_reason == 0)
        with_stop_reason_count = sum(
            1 for stop_reason in list(
                stop_replies.values()) if stop_reason != 0)

        # All but one thread should report no stop reason.
        triple = self.dbg.GetSelectedPlatform().GetTriple()

        # Consider one more thread created by calling DebugBreakProcess.
        if re.match(".*-.*-windows", triple):
            self.assertGreaterEqual(no_stop_reason_count, self.THREAD_COUNT - 1)
        else:
            self.assertEqual(no_stop_reason_count, self.THREAD_COUNT - 1)

        # Only one thread should should indicate a stop reason.
        self.assertEqual(with_stop_reason_count, 1)
