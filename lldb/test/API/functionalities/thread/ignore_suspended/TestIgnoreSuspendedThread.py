"""
Test that suspended threads do not affect should-stop decisions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class IgnoreSuspendedThreadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        #Call super's setUp().
        TestBase.setUp(self)
        #Find the line numbers for our breakpoints.
        self.break_1 = line_number('main.cpp', '// Set first breakpoint here')
        self.break_2 = line_number('main.cpp', '// Set second breakpoint here')
        self.break_3 = line_number('main.cpp', '// Set third breakpoint here')

    def printThreadsStoppedByBreakpoint(self, process):
        stopped_threads = \
            lldbutil.get_stopped_threads(process, lldb.eStopReasonBreakpoint)
        for thread in stopped_threads:
            break_id = thread.GetStopReasonDataAtIndex(0)
            print('Thread ' + str(thread.GetThreadID()) + \
                    ' stopped at breakpoint ' + str(break_id))

    def test(self):
        self.build()
        target  = lldbutil.run_to_breakpoint_make_target(self)

        #This should create a breakpoint with 1 location.
        bp1_id = \
            lldbutil.run_break_set_by_file_and_line(self,
                                                    "main.cpp",
                                                    self.break_1,
                                                    num_expected_locations=1)

        bp2_id = \
            lldbutil.run_break_set_by_file_and_line(self,
                                                    "main.cpp",
                                                    self.break_2,
                                                    num_expected_locations=1)

        bp3_id = \
            lldbutil.run_break_set_by_file_and_line(self,
                                                    "main.cpp",
                                                    self.break_3,
                                                    num_expected_locations=1)

        #Run the program.
        self.runCmd("run", RUN_SUCCEEDED)
        #Get the target process
        process = target.GetProcess()

        if self.TraceOn():
            print('First stop:')
            self.printThreadsStoppedByBreakpoint(process)

        thread_to_suspend = \
            lldbutil.get_one_thread_stopped_at_breakpoint_id(process,
                                                                bp1_id)
        self.assertIsNotNone(thread_to_suspend, "Should hit breakpoint 1")
        thread_to_suspend.Suspend()

        #Do not stop at bp2 and autocontinue to bp3
        target.FindBreakpointByID(bp2_id).SetAutoContinue(True)

        #Run to the third breakpoint
        self.runCmd("continue")

        if self.TraceOn():
            print('Second stop:')
            self.printThreadsStoppedByBreakpoint(process)

        stopped_thread = \
            lldbutil.get_one_thread_stopped_at_breakpoint_id(process,
                                                                bp3_id)
        self.assertIsNotNone(stopped_thread,
                             "Should hit breakpoint 3")

        thread_to_suspend.Resume()

        #Run to completion
        self.runCmd("continue")

        #At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(),
                            lldb.eStateExited,
                                PROCESS_EXITED)
