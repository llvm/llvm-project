"""
Test number of threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ExitDuringStepTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread", "exit_during_step")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_with_dsym(self):
        """Test thread exit during step handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.exit_during_step_inst_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_step_over_with_dsym(self):
        """Test thread exit during step-over handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.exit_during_step_over_test()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_step_in_with_dsym(self):
        """Test thread exit during step-in handling."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.exit_during_step_in_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_with_dwarf(self):
        """Test thread exit during step handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.exit_during_step_inst_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_step_over_with_dwarf(self):
        """Test thread exit during step-over handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.exit_during_step_over_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_step_in_with_dwarf(self):
        """Test thread exit during step-in handling."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.exit_during_step_in_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')
        self.continuepoint = line_number('main.cpp', '// Continue from here')

    def exit_during_step_inst_test(self):
        """Test thread exit while using step-inst."""
        self.exit_during_step_base("thread step-inst -m all-threads", 'stop reason = instruction step')

    def exit_during_step_over_test(self):
        """Test thread exit while using step-over."""
        self.exit_during_step_base("thread step-over -m all-threads", 'stop reason = step over')

    def exit_during_step_in_test(self):
        """Test thread exit while using step-in."""
        self.exit_during_step_base("thread step-in -m all-threads", 'stop reason = step in')

    def exit_during_step_base(self, step_cmd, step_stop_reason):
        """Test thread exit during step handling."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
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

        # Make sure we see all three threads
        self.assertTrue(num_threads == 3, 'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        thread1 = process.GetThreadAtIndex(0)
        thread2 = process.GetThreadAtIndex(1)
        thread3 = process.GetThreadAtIndex(2)

        # Make sure all threads are stopped
        self.assertTrue(thread1.IsStopped(), "Thread 1 didn't stop during breakpoint")
        self.assertTrue(thread2.IsStopped(), "Thread 2 didn't stop during breakpoint")
        self.assertTrue(thread3.IsStopped(), "Thread 3 didn't stop during breakpoint")

        # Keep stepping until we've reached our designated continue point
        stepping_thread = process.GetSelectedThread()
        current_line = self.breakpoint
        stepping_frame = stepping_thread.GetFrameAtIndex(0)
        self.assertTrue(current_line == stepping_frame.GetLineEntry().GetLine(), "Starting line for stepping doesn't match breakpoint line.")
        while current_line != self.continuepoint:
            self.runCmd(step_cmd)

            if stepping_thread != process.GetSelectedThread():
                process.SetSelectedThread(stepping_thread)

            frame = stepping_thread.GetFrameAtIndex(0)

            current_line = frame.GetLineEntry().GetLine()

            self.assertTrue(current_line >= self.breakpoint, "Stepped to unexpected line, " + str(current_line))
            self.assertTrue(current_line <= self.continuepoint, "Stepped to unexpected line, " + str(current_line))

        self.runCmd("thread list")

        # Update the number of threads
        num_threads = process.GetNumThreads()

        # Check to see that we reduced the number of threads as expected
        self.assertTrue(num_threads == 2, 'Number of expected threads and actual threads do not match after thread exit.')

        self.expect("thread list", 'Process state is stopped due to step',
                substrs = ['stopped',
                           step_stop_reason])

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
