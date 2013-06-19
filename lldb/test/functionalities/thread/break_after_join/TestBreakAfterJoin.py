"""
Test number of threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class BreakpointAfterJoinTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread", "break_after_join")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @dsym_test
    def test_with_dsym(self):
        """Test breakpoint handling after a thread join."""
        self.buildDsym(dictionary=self.getBuildFlags())
        self.breakpoint_after_join_test()

    @expectedFailureDarwin("llvm.org/pr15824") # thread states not properly maintained
    @skipIfLinux # Causes hangs (llvm.org/pr16170) when run using "make check"
    @dwarf_test
    def test_with_dwarf(self):
        """Test breakpoint handling after a thread join."""
        self.buildDwarf(dictionary=self.getBuildFlags())
        self.breakpoint_after_join_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')

    def breakpoint_after_join_test(self):
        """Test breakpoint handling after a thread join."""
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

        # The exit probably occured during breakpoint handling, but it isn't
        # guaranteed.  The main thing we're testing here is that the debugger
        # handles this cleanly is some way.

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see six threads
        self.assertTrue(num_threads == 6, 'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        thread1 = process.GetThreadAtIndex(0)
        thread2 = process.GetThreadAtIndex(1)
        thread3 = process.GetThreadAtIndex(2)
        thread4 = process.GetThreadAtIndex(3)
        thread5 = process.GetThreadAtIndex(4)
        thread6 = process.GetThreadAtIndex(5)

        # Make sure all threads are stopped
        self.assertTrue(thread1.IsStopped(), "Thread 1 didn't stop during breakpoint")
        self.assertTrue(thread2.IsStopped(), "Thread 2 didn't stop during breakpoint")
        self.assertTrue(thread3.IsStopped(), "Thread 3 didn't stop during breakpoint")
        self.assertTrue(thread4.IsStopped(), "Thread 4 didn't stop during breakpoint")
        self.assertTrue(thread5.IsStopped(), "Thread 5 didn't stop during breakpoint")
        self.assertTrue(thread6.IsStopped(), "Thread 6 didn't stop during breakpoint")

        # Run to completion
        self.runCmd("continue")

        # If the process hasn't exited, collect some information
        if process.GetState() != lldb.eStateExited:
            self.runCmd("thread list")
            self.runCmd("process status")

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
