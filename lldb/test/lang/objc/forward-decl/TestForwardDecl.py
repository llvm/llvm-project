"""Test that a forward-declared class works when its complete definition is in a library"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ForwardDeclTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "forward-decl")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_expr_with_dsym(self):
        self.buildDsym()
        self.expr()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_expr_with_dwarf(self):
        self.buildDwarf()
        self.expr()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

    def expr(self):
        self.common_setup()

        # This should display correctly.
        self.expect("expression [j getMember]", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 0x"])
            
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
