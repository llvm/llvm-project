"""
Test that we page getting a long backtrace on more than one thread
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestThreadBacktracePage(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_thread_backtrace_one_thread(self):
        """Run a simplified version of the test that just hits one breakpoint and
        doesn't care about synchronizing the two threads - hopefully this will
        run on more systems."""
        self.build()
        (
            self.inferior_target,
            self.process,
            thread,
            bkpt,
        ) = lldbutil.run_to_source_breakpoint(
            self, self.bkpt_string, lldb.SBFileSpec("main.cpp"), only_one_thread=False
        )

        # We hit the breakpoint on at least one thread.  If we hit it on both threads
        # simultaneously, we are ready to run our tests.  Otherwise, suspend the thread
        # that hit the breakpoint, and continue till the second thread hits
        # the breakpoint:

        (breakpoint_threads, other_threads) = ([], [])
        lldbutil.sort_stopped_threads(
            self.process,
            breakpoint_threads=breakpoint_threads,
            other_threads=other_threads,
        )
        self.assertGreater(
            len(breakpoint_threads), 0, "We hit at least one breakpoint thread"
        )
        self.assertGreater(len(breakpoint_threads[0].frames), 2, "I can go up")
        thread_id = breakpoint_threads[0].idx
        name = breakpoint_threads[0].frame[1].name.split("(")[0]
        self.check_one_thread(thread_id, name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.bkpt_string = "// Set breakpoint here"

    def check_one_thread(self, thread_id, func_name):
        # Now issue some thread backtrace commands and make sure they
        # get the right answer back.
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Run the real backtrace, remember to pass True for add_to_history since
        # we don't generate repeat commands for commands that aren't going into the history.
        interp.HandleCommand(
            "thread backtrace --count 10 {0}".format(thread_id), result, True
        )
        self.assertTrue(result.Succeeded(), "bt with count succeeded")
        # There should be 11 lines:
        lines = result.GetOutput().splitlines()
        self.assertEqual(len(lines), 11, "Got the right number of lines")
        # First frame is stop_here:
        self.assertNotEqual(lines[1].find("stop_here"), -1, "Found Stop Here")
        for line in lines[2:10]:
            self.assertNotEqual(
                line.find(func_name),
                -1,
                "Name {0} not found in line: {1}".format(func_name, line),
            )
        # The last entry should be 43:
        self.assertNotEqual(lines[10].find("count=43"), -1, "First show ends at 43")

        # Now try a repeat, and make sure we get 10 more on this thread:
        # import pdb; pdb.set_trace()
        interp.HandleCommand("", result, True)
        self.assertTrue(
            result.Succeeded(), "repeat command failed: {0}".format(result.GetError())
        )
        lines = result.GetOutput().splitlines()
        self.assertEqual(len(lines), 11, "Repeat got 11 lines")
        # Every line should now be the recurse function:
        for line in lines[1:10]:
            self.assertNotEqual(line.find(func_name), -1, "Name in every line")
        self.assertNotEqual(lines[10].find("count=33"), -1, "Last one is now 33")

    def check_two_threads(
        self, result_str, thread_id_1, name_1, thread_id_2, name_2, start_idx, end_idx
    ):
        # We should have 2 occurrences ot the thread header:
        self.assertEqual(
            result_str.count("thread #{0}".format(thread_id_1)), 1, "One for thread 1"
        )
        self.assertEqual(
            result_str.count("thread #{0}".format(thread_id_2)), 1, "One for thread 2"
        )
        # We should have 10 occurrences of each name:
        self.assertEqual(result_str.count(name_1), 10, "Found 10 of {0}".format(name_1))
        self.assertEqual(result_str.count(name_2), 10, "Found 10 of {0}".format(name_1))
        # There should be two instances of count=<start_idx> and none of count=<start-1>:
        self.assertEqual(
            result_str.count("count={0}".format(start_idx)),
            2,
            "Two instances of start_idx",
        )
        self.assertEqual(
            result_str.count("count={0}".format(start_idx - 1)),
            0,
            "No instances of start_idx - 1",
        )
        # There should be two instances of count=<end_idx> and none of count=<end_idx+1>:
        self.assertEqual(
            result_str.count("count={0}".format(end_idx)), 2, "Two instances of end_idx"
        )
        self.assertEqual(
            result_str.count("count={0}".format(end_idx + 1)),
            0,
            "No instances after end idx",
        )

    # The setup of this test was copied from the step-out test, and I can't tell from
    # the comments whether it was getting two threads to the same breakpoint that was
    # problematic, or the step-out part.  This test stops at the rendevous point so I'm
    # removing the skipIfLinux to see if we see any flakiness in just this part of the test.
    @skipIfWindows  # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureNetBSD
    def test_thread_backtrace_two_threads(self):
        """Test that repeat works even when backtracing on more than one thread."""
        self.build()
        (
            self.inferior_target,
            self.process,
            thread,
            bkpt,
        ) = lldbutil.run_to_source_breakpoint(
            self, self.bkpt_string, lldb.SBFileSpec("main.cpp"), only_one_thread=False
        )

        # We hit the breakpoint on at least one thread.  If we hit it on both threads
        # simultaneously, we are ready to run our tests.  Otherwise, suspend the thread
        # that hit the breakpoint, and continue till the second thread hits
        # the breakpoint:

        (breakpoint_threads, other_threads) = ([], [])
        lldbutil.sort_stopped_threads(
            self.process,
            breakpoint_threads=breakpoint_threads,
            other_threads=other_threads,
        )
        if len(breakpoint_threads) == 1:
            success = thread.Suspend()
            self.assertTrue(success, "Couldn't suspend a thread")
            breakpoint_threads = lldbutil.continue_to_breakpoint(self.process, bkpt)
            self.assertEqual(len(breakpoint_threads), 2, "Second thread stopped")

        # Figure out which thread is which:
        thread_id_1 = breakpoint_threads[0].idx
        self.assertGreater(len(breakpoint_threads[0].frames), 2, "I can go up")
        name_1 = breakpoint_threads[0].frame[1].name.split("(")[0]

        thread_id_2 = breakpoint_threads[1].idx
        self.assertGreater(len(breakpoint_threads[1].frames), 2, "I can go up")
        name_2 = breakpoint_threads[1].frame[1].name.split("(")[0]

        # Check that backtrace and repeat works on one thread, then works on the second
        # when we switch to it:
        self.check_one_thread(thread_id_1, name_1)
        self.check_one_thread(thread_id_2, name_2)

        # The output is looking right at this point, let's just do a couple more quick checks
        # to see we handle two threads and a start count:
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        interp.HandleCommand(
            "thread backtrace --count 10 --start 10 {0} {1}".format(
                thread_id_1, thread_id_2
            ),
            result,
            True,
        )
        self.assertTrue(result.Succeeded(), "command succeeded for two threads")

        result.Clear()
        interp.HandleCommand("", result, True)
        self.assertTrue(result.Succeeded(), "repeat command succeeded for two threads")
        result_str = result.GetOutput()
        self.check_two_threads(
            result_str, thread_id_1, name_1, thread_id_2, name_2, 23, 32
        )

        # Finally make sure the repeat repeats:
        result.Clear()
        interp.HandleCommand("", result, True)
        self.assertTrue(result.Succeeded(), "repeat command succeeded for two threads")
        result_str = result.GetOutput()
        self.check_two_threads(
            result_str, thread_id_1, name_1, thread_id_2, name_2, 13, 22
        )
