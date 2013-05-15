"""
Test number of threads.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class NumberOfThreadsTestCase(TestBase):

    mydir = os.path.join("functionalities", "thread")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test number of threads."""
        self.buildDsym()
        self.number_of_threads_test()

    @dwarf_test
    def test_with_dwarf(self):
        """Test number of threads."""
        self.buildDwarf()
        self.number_of_threads_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def number_of_threads_test(self):
        """Test number of threads."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 location.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1)

        # The breakpoint list should show 3 locations.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file ='main.c', line = %d, locations = 1" % self.line])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        self.assertTrue(num_threads == 4, 'Number of expected threads and actual threads do not match.')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
