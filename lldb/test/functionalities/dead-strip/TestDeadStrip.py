"""
Test that breakpoint works correctly in the presence of dead-code stripping.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class DeadStripTestCase(TestBase):

    mydir = os.path.join("functionalities", "dead-strip")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test breakpoint works correctly with dead-code stripping."""
        self.buildDsym()
        self.dead_strip()

    @dwarf_test
    def test_with_dwarf(self):
        """Test breakpoint works correctly with dead-code stripping."""
        self.buildDwarf()
        self.dead_strip()

    def dead_strip(self):
        """Test breakpoint works correctly with dead-code stripping."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name f1 (live code).
        lldbutil.run_break_set_by_symbol (self, "f1", extra_options="-s a.out", num_expected_locations=1, module_name="a.out")

        # Break by function name f2 (dead code).
        lldbutil.run_break_set_by_symbol (self, "f2", extra_options="-s a.out", num_expected_locations=0)

        # Break by function name f3 (live code).
        lldbutil.run_break_set_by_symbol (self, "f3", extra_options="-s a.out", num_expected_locations=1, module_name="a.out")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint (breakpoint #1).
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'a.out`f1',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f 1", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint (breakpoint #3).
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'a.out`f3',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f 3", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
