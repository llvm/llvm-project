import json
import re

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemote_vContThreads(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def start_threads(self, num):
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=['thread:new'] * num + ['@started'])
        # start the process and wait for output
        self.test_sequence.add_log_lines([
            "read packet: $c#63",
            {"type": "output_match", "regex": self.maybe_strict_output_regex(
                r"@started\r\n")},
        ], True)
        # then interrupt it
        self.add_interrupt_packets()
        self.add_threadinfo_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)
        self.assertEqual(len(threads), num + 1)

        self.reset_test_sequence()
        return threads

    @skipUnlessPlatform(["netbsd"])
    @expectedFailureNetBSD
    def test_signal_one_thread(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending a signal to one of the two threads
        self.test_sequence.add_log_lines([
            "read packet: $vCont;C{0:x}:{1:x};c#00".format(
                lldbutil.get_signal_number('SIGUSR1'), threads[0]),
            {"direction": "send", "regex": r"^\$W00#b7$"},
        ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipUnlessPlatform(["netbsd"])
    @expectedFailureNetBSD
    def test_signal_all_threads(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending a signal to two threads (= the process)
        self.test_sequence.add_log_lines([
            "read packet: $vCont;C{0:x}:{1:x};C{0:x}:{2:x}#00".format(
                lldbutil.get_signal_number('SIGUSR1'),
                threads[0], threads[1]),
            {"direction": "send", "regex": r"^\$W00#b7$"},
        ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipUnlessPlatform(["netbsd"])
    def test_signal_two_of_three_threads(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(2)
        # try sending a signal to 2 out of 3 threads
        self.test_sequence.add_log_lines([
            "read packet: $vCont;C{0:x}:{1:x};C{0:x}:{2:x};c#00".format(
                lldbutil.get_signal_number('SIGUSR1'),
                threads[1], threads[2]),
            {"direction": "send", "regex": r"^\$E1e#db$"},
        ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipUnlessPlatform(["netbsd"])
    def test_signal_two_signals(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending two different signals to two threads
        self.test_sequence.add_log_lines([
            "read packet: $vCont;C{0:x}:{1:x};C{2:x}:{3:x}#00".format(
                lldbutil.get_signal_number('SIGUSR1'), threads[0],
                lldbutil.get_signal_number('SIGUSR2'), threads[1]),
            {"direction": "send", "regex": r"^\$E1e#db$"},
        ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
