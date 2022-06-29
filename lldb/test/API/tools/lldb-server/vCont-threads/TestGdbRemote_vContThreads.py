import json
import re
import time

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemote_vContThreads(gdbremote_testcase.GdbRemoteTestCaseBase):

    def start_threads(self, num):
        procs = self.prep_debug_monitor_and_inferior(inferior_args=[str(num)])
        # start the process and wait for output
        self.test_sequence.add_log_lines([
            "read packet: $c#63",
            {"type": "output_match", "regex": r".*@started\r\n.*"},
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

    def send_and_check_signal(self, vCont_data, threads):
        self.test_sequence.add_log_lines([
            "read packet: $vCont;{0}#00".format(vCont_data),
            {"type": "output_match",
             "regex": len(threads) *
                      r".*received SIGUSR1 on thread id: ([0-9a-f]+)\r\n.*",
             "capture": dict((i, "tid{0}".format(i)) for i
                             in range(1, len(threads)+1)),
             },
        ], True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        tids = sorted(int(context["tid{0}".format(x)], 16)
                      for x in range(1, len(threads)+1))
        self.assertEqual(tids, sorted(threads))

    def get_pid(self):
        self.add_process_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        procinfo = self.parse_process_info_response(context)
        return int(procinfo['pid'], 16)

    @skipIfWindows
    @skipIfDarwin
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_process_without_tid(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}".format(lldbutil.get_signal_number('SIGUSR1')),
            threads)

    @skipUnlessPlatform(["netbsd"])
    @expectedFailureNetBSD
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_one_thread(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending a signal to one of the two threads
        self.send_and_check_signal(
            "C{0:x}:{1:x};c".format(lldbutil.get_signal_number('SIGUSR1')),
            threads[:1])

    @skipIfWindows
    @skipIfDarwin
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_all_threads(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending a signal to two threads (= the process)
        self.send_and_check_signal(
            "C{0:x}:{1:x};C{0:x}:{2:x}".format(
                lldbutil.get_signal_number('SIGUSR1'),
                *threads),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_process_by_pid(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}:p{1:x}".format(
                lldbutil.get_signal_number('SIGUSR1'),
                self.get_pid()),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_process_minus_one(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}:p-1".format(
                lldbutil.get_signal_number('SIGUSR1')),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_minus_one(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}:-1".format(lldbutil.get_signal_number('SIGUSR1')),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_all_threads_by_pid(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        # try sending a signal to two threads (= the process)
        self.send_and_check_signal(
            "C{0:x}:p{1:x}.{2:x};C{0:x}:p{1:x}.{3:x}".format(
                lldbutil.get_signal_number('SIGUSR1'),
                self.get_pid(),
                *threads),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_minus_one_by_pid(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}:p{1:x}.-1".format(
                lldbutil.get_signal_number('SIGUSR1'),
                self.get_pid()),
            threads)

    @skipIfWindows
    @expectedFailureNetBSD
    @expectedFailureAll(oslist=["freebsd"],
                        bugnumber="github.com/llvm/llvm-project/issues/56086")
    @add_test_categories(["llgs"])
    @skipIfAsan # Times out under asan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_signal_minus_one_by_minus_one(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = self.start_threads(1)
        self.send_and_check_signal(
            "C{0:x}:p-1.-1".format(
                lldbutil.get_signal_number('SIGUSR1')),
            threads)

    @skipUnlessPlatform(["netbsd"])
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
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
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
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

    THREAD_MATCH_RE = re.compile(r"thread ([0-9a-f]+) running")

    def continue_and_get_threads_running(self, continue_packet):
        self.test_sequence.add_log_lines(
            ["read packet: ${}#00".format(continue_packet),
             ], True)
        self.expect_gdbremote_sequence()
        self.reset_test_sequence()
        time.sleep(1)
        self.add_interrupt_packets()
        exp = self.expect_gdbremote_sequence()
        found = set()
        for line in exp["O_content"].decode().splitlines():
            m = self.THREAD_MATCH_RE.match(line)
            if m is not None:
                found.add(int(m.group(1), 16))
        return found

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_run_subset_of_threads(self):
        self.build()
        self.set_inferior_startup_launch()

        threads = set(self.start_threads(3))
        all_subthreads = self.continue_and_get_threads_running("c")
        all_subthreads_list = list(all_subthreads)
        self.assertEqual(len(all_subthreads), 3)
        self.assertEqual(threads & all_subthreads, all_subthreads)

        # resume two threads explicitly, stop the third one implicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;c:{:x};c:{:x}".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[:2]))

        # resume two threads explicitly, stop others explicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;c:{:x};c:{:x};t".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[:2]))

        # stop one thread explicitly, resume others
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;t:{:x};c".format(all_subthreads_list[-1])),
            set(all_subthreads_list[:2]))

        # resume one thread explicitly, stop one explicitly,
        # resume others
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;c:{:x};t:{:x};c".format(*all_subthreads_list[-2:])),
            set(all_subthreads_list[:2]))

        # resume one thread explicitly, stop one explicitly,
        # stop others implicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;t:{:x};c:{:x}".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[1:2]))

        # resume one thread explicitly, stop one explicitly,
        # stop others explicitly
        self.assertEqual(
            self.continue_and_get_threads_running(
                "vCont;t:{:x};c:{:x};t".format(*all_subthreads_list[:2])),
            set(all_subthreads_list[1:2]))
