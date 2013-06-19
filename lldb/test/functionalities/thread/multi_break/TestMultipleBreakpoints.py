"""
Test number of threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class MultipleBreakpointTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread", "multi_break")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_with_dsym(self):
        """Test simultaneous breakpoints in multiple threads."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.multiple_breakpoint_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dwarf_test
    def test_with_dwarf(self):
        """Test simultaneous breakpoints in multiple threads."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.multiple_breakpoint_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')

    def multiple_breakpoint_test(self):
        """Test simultaneous breakpoints in multiple threads."""
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
        # The breakpoint may be hit in either thread 2 or thread 3.
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

        # Make sure both threads are stopped
        self.assertTrue(thread1.IsStopped(), "Primary thread didn't stop during breakpoint")
        self.assertTrue(thread2.IsStopped(), "Secondary thread didn't stop during breakpoint")
        self.assertTrue(thread3.IsStopped(), "Tertiary thread didn't stop during breakpoint")

        # Delete the first breakpoint then continue
        self.runCmd("breakpoint delete 1")

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
