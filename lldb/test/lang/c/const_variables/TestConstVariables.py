"""Check that compiler-generated constant values work correctly"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ConstVariableTestCase(TestBase):

    mydir = os.path.join("lang", "c", "const_variables")

    @dsym_test
    @unittest2.expectedFailure(13314878)
    def test_with_dsym_and_run_command(self):
        """Test interpreted and JITted expressions on constant values."""
        self.buildDsym()
        self.const_variable()

    @skipIfLinux # This test works with gcc, but fails with newer version of clang on Linux due to a clang issue.  Bug number TDB.
    @dwarf_test
    @unittest2.expectedFailure(13314878)
    def test_with_dwarf_and_run_command(self):
        """Test interpreted and JITted expressions on constant values."""
        self.buildDwarf()
        self.const_variable()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def const_variable(self):
        """Test interpreted and JITted expressions on constant values."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("next")

        # Try frame variable.
        self.expect("frame variable index", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int32_t) index = 512'])

        # Try an interpreted expression.
        self.expect("expr (index + 512)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) $0 = 1024'])

        # Try a JITted expression.
        self.expect("expr (int)getpid(); (index - 256)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int) $1 = 256'])

        self.runCmd("kill")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
