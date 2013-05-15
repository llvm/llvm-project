"""
Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class Radar12481949DataFormatterTestCase(TestBase):

    # test for rdar://problem/12481949
    mydir = os.path.join("python_api", "rdar-12481949")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1."""
        self.buildDsym()
        self.rdar12481949_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1."""
        self.buildDwarf()
        self.rdar12481949_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def rdar12481949_commands(self):
        """Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format delete hex', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsSigned() == -1, "GetValueAsSigned() says -1")
        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsSigned() != 0xFFFFFFFF, "GetValueAsSigned() does not say 0xFFFFFFFF")
        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsSigned() != 0xFFFFFFFFFFFFFFFF, "GetValueAsSigned() does not say 0xFFFFFFFFFFFFFFFF")

        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsUnsigned() != -1, "GetValueAsUnsigned() does not say -1")
        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsUnsigned() == 0xFFFFFFFFFFFFFFFF, "GetValueAsUnsigned() says 0xFFFFFFFFFFFFFFFF")
        self.assertTrue(self.frame().FindVariable("myvar").GetValueAsSigned() != 0xFFFFFFFF, "GetValueAsUnsigned() does not say 0xFFFFFFFF")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
