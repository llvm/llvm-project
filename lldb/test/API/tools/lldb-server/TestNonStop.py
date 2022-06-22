from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

import gdbremote_testcase


class LldbGdbServerTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # no SIGSEGV support
    @add_test_categories(["llgs"])
    def test_run(self):
        self.build()
        self.set_inferior_startup_launch()
        thread_num = 3
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:segfault"] + thread_num * ["thread:new"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()

        segv_signo = lldbutil.get_signal_number('SIGSEGV')
        all_threads = set()
        all_segv_threads = []

        # we should get segfaults from all the threads
        for segv_no in range(thread_num):
            # first wait for the notification event
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [{"direction": "send",
                  "regex": r"^%Stop:(T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);)",
                  "capture": {1: "packet", 2: "signo", 3: "thread_id"},
                  },
                 ], True)
            m = self.expect_gdbremote_sequence()
            del m["O_content"]
            threads = [m]

            # then we may get events for the remaining threads
            # (but note that not all threads may have been started yet)
            while True:
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    ["read packet: $vStopped#00",
                     {"direction": "send",
                      "regex": r"^\$(OK|T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);)",
                      "capture": {1: "packet", 2: "signo", 3: "thread_id"},
                      },
                     ], True)
                m = self.expect_gdbremote_sequence()
                if m["packet"] == "OK":
                    break
                del m["O_content"]
                threads.append(m)

            segv_threads = []
            other_threads = []
            for t in threads:
                signo = int(t["signo"], 16)
                if signo == segv_signo:
                    segv_threads.append(t["thread_id"])
                else:
                    self.assertEqual(signo, 0)
                    other_threads.append(t["thread_id"])

            # verify that exactly one thread segfaulted
            self.assertEqual(len(segv_threads), 1)
            # we should get only one segv from every thread
            self.assertNotIn(segv_threads[0], all_segv_threads)
            all_segv_threads.extend(segv_threads)
            # segv_threads + other_threads should always be a superset
            # of all_threads, i.e. we should get states for all threads
            # already started
            self.assertFalse(
                    all_threads.difference(other_threads + segv_threads))
            all_threads.update(other_threads + segv_threads)

            # verify that `?` returns the same result
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $?#00",
                 ], True)
            threads_verify = []
            while True:
                self.test_sequence.add_log_lines(
                    [{"direction": "send",
                      "regex": r"^\$(OK|T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);)",
                      "capture": {1: "packet", 2: "signo", 3: "thread_id"},
                      },
                     ], True)
                m = self.expect_gdbremote_sequence()
                if m["packet"] == "OK":
                    break
                del m["O_content"]
                threads_verify.append(m)
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    ["read packet: $vStopped#00",
                     ], True)

            self.assertEqual(threads, threads_verify)

            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vCont;C{:02x}:{};c#00"
                 .format(segv_signo, segv_threads[0]),
                 "send packet: $OK#00",
                 ], True)
            self.expect_gdbremote_sequence()

        # finally, verify that all threads have started
        self.assertEqual(len(all_threads), thread_num + 1)

    @add_test_categories(["llgs"])
    def test_vCtrlC(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "read packet: $vCtrlC#00",
             "send packet: $OK#00",
             {"direction": "send",
              "regex": r"^%Stop:T",
              },
             ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_exit(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "send packet: %Stop:W00#00",
             "read packet: $vStopped#00",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()

    @skipIfWindows  # no clue, the result makes zero sense
    @add_test_categories(["llgs"])
    def test_exit_query(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "send packet: %Stop:W00#00",
             "read packet: $?#00",
             "send packet: $W00#00",
             "read packet: $vStopped#00",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()
