"""
Test number of threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class CreateDuringStepTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread", "create_during_step")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_step_inst_with_dsym(self):
        """Test thread creation during step-inst handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.create_during_step_inst_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_step_over_with_dsym(self):
        """Test thread creation during step-over handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.create_during_step_over_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_step_in_with_dsym(self):
        """Test thread creation during step-in handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.create_during_step_in_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_step_inst_with_dwarf(self):
        """Test thread creation during step-inst handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.create_during_step_inst_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_step_over_with_dwarf(self):
        """Test thread creation during step-over handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.create_during_step_over_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_step_in_with_dwarf(self):
        """Test thread creation during step-in handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.create_during_step_in_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')
        self.continuepoint = line_number('main.cpp', '// Continue from here')

    def create_during_step_inst_test(self):
        """Test thread creation while using step-inst."""
        self.create_during_step_base("thread step-inst -m all-threads", 'stop reason = instruction step')

    def create_during_step_over_test(self):
        """Test thread creation while using step-over."""
        self.create_during_step_base("thread step-over -m all-threads", 'stop reason = step over')

    def create_during_step_in_test(self):
        """Test thread creation while using step-in."""
        self.create_during_step_base("thread step-in -m all-threads", 'stop reason = step in')

    def create_during_step_base(self, step_cmd, step_stop_reason):
        """Test thread creation while using step-in."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the stepping thread.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # The breakpoint list should show 1 location.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file ='main.cpp', line = %d, locations = 1" % self.breakpoint])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see only two threads
        self.assertTrue(num_threads == 2, 'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        thread1 = process.GetThreadAtIndex(0)
        thread2 = process.GetThreadAtIndex(1)

        # Make sure both threads are stopped
        self.assertTrue(thread1.IsStopped(), "Thread 1 didn't stop during breakpoint")
        self.assertTrue(thread2.IsStopped(), "Thread 2 didn't stop during breakpoint")

        # Keep stepping until we've reached our designated continue point
        stepping_thread = process.GetSelectedThread()
        current_line = self.breakpoint
        while current_line != self.continuepoint:
            self.runCmd(step_cmd)

            # The thread creation may change the selected thread.
            # If it does, we just change it back here.
            if stepping_thread != process.GetSelectedThread():
                process.SetSelectedThread(stepping_thread)

            frame = stepping_thread.GetFrameAtIndex(0)
            current_line = frame.GetLineEntry().GetLine()

            # Make sure we're still where we thought we were
            self.assertTrue(current_line >= self.breakpoint, "Stepped to unexpected line, " + str(current_line))
            self.assertTrue(current_line <= self.continuepoint, "Stepped to unexpected line, " + str(current_line))

        # Update the number of threads
        num_threads = process.GetNumThreads()

        # Check to see that we increased the number of threads as expected
        self.assertTrue(num_threads == 3, 'Number of expected threads and actual threads do not match after thread exit.')

        self.expect("thread list", 'Process state is stopped due to step',
                substrs = ['stopped',
                           step_stop_reason])

        # Run to completion
        self.runCmd("process continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
