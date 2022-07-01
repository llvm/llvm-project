import re

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestPartialResume(gdbremote_testcase.GdbRemoteTestCaseBase):
    THREAD_MATCH_RE = re.compile(r"thread ([0-9a-f]+) running")

    def start_vCont_run_subset_of_threads_test(self):
        self.build()
        self.set_inferior_startup_launch()

        procs = self.prep_debug_monitor_and_inferior(inferior_args=["3"])
        # grab the main thread id
        self.add_threadinfo_collection_packets()
        main_thread = self.parse_threadinfo_packets(
            self.expect_gdbremote_sequence())
        self.assertEqual(len(main_thread), 1)
        self.reset_test_sequence()

        # run until threads start, then grab full thread list
        self.test_sequence.add_log_lines([
            "read packet: $c#63",
            {"direction": "send", "regex": "[$]T.*;reason:signal.*"},
        ], True)
        self.add_threadinfo_collection_packets()

        all_threads = self.parse_threadinfo_packets(
            self.expect_gdbremote_sequence())
        self.assertEqual(len(all_threads), 4)
        self.assertIn(main_thread[0], all_threads)
        self.reset_test_sequence()

        all_subthreads = set(all_threads) - set(main_thread)
        self.assertEqual(len(all_subthreads), 3)

        return (main_thread[0], list(all_subthreads))

    def continue_and_get_threads_running(self, main_thread, vCont_req):
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c:{:x};{}#00".format(main_thread, vCont_req),
             "send packet: $W00#00",
             ], True)
        exp = self.expect_gdbremote_sequence()
        self.reset_test_sequence()
        found = set()
        for line in exp["O_content"].decode().splitlines():
            m = self.THREAD_MATCH_RE.match(line)
            if m is not None:
                found.add(int(m.group(1), 16))
        return found

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_cxcx(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # resume two threads explicitly, stop the third one implicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "c:{:x};c:{:x}".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[:2]))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_cxcxt(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # resume two threads explicitly, stop others explicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "c:{:x};c:{:x};t".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[:2]))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_txc(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # stop one thread explicitly, resume others
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "t:{:x};c".format(all_subthreads_list[-1])),
            set(all_subthreads_list[:2]))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_cxtxc(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # resume one thread explicitly, stop one explicitly,
        # resume others
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "c:{:x};t:{:x};c".format(*all_subthreads_list[-2:])),
            set(all_subthreads_list[:2]))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_txcx(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # resume one thread explicitly, stop one explicitly,
        # stop others implicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "t:{:x};c:{:x}".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[1:2]))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_txcxt(self):
        main_thread, all_subthreads_list = (
            self.start_vCont_run_subset_of_threads_test())
        # resume one thread explicitly, stop one explicitly,
        # stop others explicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                main_thread,
                "t:{:x};c:{:x};t".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[1:2]))
