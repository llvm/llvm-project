


import unittest2
import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemote_qThreadStopInfo(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)
    THREAD_COUNT = 5

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    @skipIfDarwinEmbedded # <rdar://problem/27005337>
    def gather_stop_replies_via_qThreadStopInfo(self, thread_count):
        # Set up the inferior args.
        inferior_args = []
        for i in range(thread_count - 1):
            inferior_args.append("thread:new")
        inferior_args.append("sleep:10")
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=inferior_args)

        # Assumes test_sequence has anything added needed to setup the initial state.
        # (Like optionally enabling QThreadsInStopReply.)
        self.test_sequence.add_log_lines([
            "read packet: $c#63"
        ], True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

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

        # On Windows, there could be more threads spawned. For example, DebugBreakProcess will
        # create a new thread from the debugged process to handle an exception event. So here we
        # assert 'GreaterEqual' condition.
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        if re.match(".*-.*-windows", triple):
            self.assertGreaterEqual(len(threads), thread_count)
        else:
            self.assertEqual(len(threads), thread_count)

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

        return (stop_replies, thread_dicts)

    def qThreadStopInfo_works_for_multiple_threads(self, thread_count):
        (stop_replies, _) = self.gather_stop_replies_via_qThreadStopInfo(thread_count)
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        # Consider one more thread created by calling DebugBreakProcess.
        if re.match(".*-.*-windows", triple):
            self.assertGreaterEqual(len(stop_replies), thread_count)
        else:
            self.assertEqual(len(stop_replies), thread_count)

    @debugserver_test
    def test_qThreadStopInfo_works_for_multiple_threads_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_works_for_multiple_threads(self.THREAD_COUNT)

    @llgs_test
    @skipIfNetBSD
    def test_qThreadStopInfo_works_for_multiple_threads_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_works_for_multiple_threads(self.THREAD_COUNT)

    def qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt(
            self, thread_count):
        (stop_replies, _) = self.gather_stop_replies_via_qThreadStopInfo(thread_count)
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
            self.assertGreaterEqual(no_stop_reason_count, thread_count - 1)
        else:
            self.assertEqual(no_stop_reason_count, thread_count - 1)

        # Only one thread should should indicate a stop reason.
        self.assertEqual(with_stop_reason_count, 1)

    @debugserver_test
    def test_qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt_debugserver(
            self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt(
            self.THREAD_COUNT)

    @expectedFailureAll(oslist=["freebsd", "netbsd"])
    @llgs_test
    def test_qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt_llgs(
            self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_only_reports_one_thread_stop_reason_during_interrupt(
            self.THREAD_COUNT)

    def qThreadStopInfo_has_valid_thread_names(
            self, thread_count, expected_thread_name):
        (_, thread_dicts) = self.gather_stop_replies_via_qThreadStopInfo(thread_count)
        self.assertIsNotNone(thread_dicts)

        for thread_dict in list(thread_dicts.values()):
            name = thread_dict.get("name")
            self.assertIsNotNone(name)
            self.assertEqual(name, expected_thread_name)

    @unittest2.skip("MacOSX doesn't have a default thread name")
    @debugserver_test
    def test_qThreadStopInfo_has_valid_thread_names_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_has_valid_thread_names(self.THREAD_COUNT, "a.out")

    # test requires OS with set, equal thread names by default.
    # Windows thread does not have name property, equal names as the process's by default.
    @skipUnlessPlatform(["linux", "windows"])
    @llgs_test
    def test_qThreadStopInfo_has_valid_thread_names_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadStopInfo_has_valid_thread_names(self.THREAD_COUNT, "a.out")
