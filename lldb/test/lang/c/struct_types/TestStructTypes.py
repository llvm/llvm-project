"""
Test that break on a struct declaration has no effect.

Instead, the first executable statement is set as the breakpoint.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class StructTypesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "struct_types")

    # rdar://problem/12566646
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test that break on a struct declaration has no effect."""
        self.buildDsym()
        self.struct_types()

    # rdar://problem/12566646
    @dwarf_test
    def test_with_dwarf(self):
        """Test that break on a struct declaration has no effect."""
        self.buildDwarf()
        self.struct_types()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.source = 'main.c'
        self.line = line_number(self.source, '// Set break point at this line.')
        self.first_executable_line = line_number(self.source,
                                                 '// This is the first executable statement.')
        self.return_line = line_number(self.source, '// This is the return statement.')

    def struct_types(self):
        """Test that break on a struct declaration has no effect and test structure access for zero sized arrays."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Break on the struct declration statement in main.c.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=False)
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.return_line, num_expected_locations=1, loc_exact=True)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if not process:
            self.fail("SBTarget.Launch() failed")

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # We should be stopped on the first executable statement within the
        # function where the original breakpoint was attempted.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['main.c:%d' % self.first_executable_line,
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        process.Continue()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Test zero length array access and make sure it succeeds with "frame variable"
        self.expect("frame variable pt.padding[0]",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ["pt.padding[0] = '"])
        self.expect("frame variable pt.padding[1]",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ["pt.padding[1] = '"])
        # Test zero length array access and make sure it succeeds with "expression"
        self.expect("expression -- (pt.padding[0])",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ["(char)", " = '"])

        # The padding should be an array of size 0
        self.expect("image lookup -t point_tag",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['padding[]']) # Once rdar://problem/12566646 is fixed, this should display correctly

        self.expect("expression -- &pt == (struct point_tag*)0",
                    substrs = ['false'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
