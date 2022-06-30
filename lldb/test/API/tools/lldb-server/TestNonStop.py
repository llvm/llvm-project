from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

import gdbremote_testcase


class LldbGdbServerTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

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

    def multiple_resume_test(self, second_command):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "read packet: ${}#00".format(second_command),
             "send packet: $E37#00",
             ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_multiple_C(self):
        self.multiple_resume_test("C05")

    @add_test_categories(["llgs"])
    def test_multiple_c(self):
        self.multiple_resume_test("c")

    @add_test_categories(["llgs"])
    def test_multiple_s(self):
        self.multiple_resume_test("s")

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_multiple_vCont(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new", "stop", "sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             {"direction": "send",
              "regex": r"^%Stop:T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+);",
              "capture": {1: "tid1"},
              },
             "read packet: $vStopped#63",
             {"direction": "send",
              "regex": r"^[$]T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+);",
              "capture": {1: "tid2"},
              },
             "read packet: $vStopped#63",
             "send packet: $OK#00",
             ], True)
        ret = self.expect_gdbremote_sequence()

        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c:{}#00".format(ret["tid1"]),
             "send packet: $OK#00",
             "read packet: $vCont;c:{}#00".format(ret["tid2"]),
             "send packet: $E37#00",
             ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_vCont_then_stop(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "read packet: $vCont;t#00",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()

    def vCont_then_partial_stop_test(self, run_both):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new", "stop", "sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             {"direction": "send",
              "regex": r"^%Stop:T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+);",
              "capture": {1: "tid1"},
              },
             "read packet: $vStopped#63",
             {"direction": "send",
              "regex": r"^[$]T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+);",
              "capture": {1: "tid2"},
              },
             "read packet: $vStopped#63",
             "send packet: $OK#00",
             ], True)
        ret = self.expect_gdbremote_sequence()

        self.reset_test_sequence()
        if run_both:
            self.test_sequence.add_log_lines(
                ["read packet: $vCont;c#00",
                 ], True)
        else:
            self.test_sequence.add_log_lines(
                ["read packet: $vCont;c:{}#00".format(ret["tid1"]),
                 ], True)
        self.test_sequence.add_log_lines(
            ["send packet: $OK#00",
             "read packet: $vCont;t:{}#00".format(ret["tid2"]),
             "send packet: $E03#00",
             ], True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_then_partial_stop(self):
        self.vCont_then_partial_stop_test(False)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_vCont_then_partial_stop_run_both(self):
        self.vCont_then_partial_stop_test(True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_stdio(self):
        self.build()
        self.set_inferior_startup_launch()
        # Since we can't easily ensure that lldb will send output in two parts,
        # just put a stop in the middle.  Since we don't clear vStdio,
        # the second message won't be delivered immediately.
        self.prep_debug_monitor_and_inferior(
            inferior_args=["message 1", "stop", "message 2"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             {"direction": "send", "regex": r"^%Stop:T.*"},
             "read packet: $vStopped#00",
             "send packet: $OK#00",
             "read packet: $c#63",
             "send packet: $OK#00",
             "send packet: %Stop:W00#00",
             ], True)
        ret = self.expect_gdbremote_sequence()
        self.assertIn(ret["O_content"], b"message 1\r\n")

        # Now, this is somewhat messy.  expect_gdbremote_sequence() will
        # automatically consume output packets, so we just send vStdio,
        # assume the first reply was consumed, send another one and expect
        # a non-consumable "OK" reply.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $vStdio#00",
             "read packet: $vStdio#00",
             "send packet: $OK#00",
             ], True)
        ret = self.expect_gdbremote_sequence()
        self.assertIn(ret["O_content"], b"message 2\r\n")

        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $vStopped#00",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_stop_reason_while_running(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new", "thread:new", "stop", "sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             # stop is used to synchronize starting threads
             "read packet: $c#63",
             "send packet: $OK#00",
             {"direction": "send", "regex": "%Stop:T.*"},
             "read packet: $c#63",
             "send packet: $OK#00",
             "read packet: $?#00",
             "send packet: $OK#00",
             ], True)
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_leave_nonstop(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
                inferior_args=["thread:new", "thread:new", "stop", "sleep:15"])
        self.test_sequence.add_log_lines(
            ["read packet: $QNonStop:1#00",
             "send packet: $OK#00",
             # stop is used to synchronize starting threads
             "read packet: $c#63",
             "send packet: $OK#00",
             {"direction": "send", "regex": "%Stop:T.*"},
             "read packet: $c#63",
             "send packet: $OK#00",
             # verify that the threads are running now
             "read packet: $?#00",
             "send packet: $OK#00",
             "read packet: $QNonStop:0#00",
             "send packet: $OK#00",
             # we should issue some random request now to verify that the stub
             # did not send stop reasons -- we may verify whether notification
             # queue was cleared while at it
             "read packet: $vStopped#00",
             "send packet: $Eff#00",
             "read packet: $?#00",
             {"direction": "send", "regex": "[$]T.*"},
             ], True)
        self.expect_gdbremote_sequence()
