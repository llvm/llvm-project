"""Test that hidden ivars in a shared library are visible from the main executable."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class HiddenIvarsTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "hidden-ivars")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_expr_with_dsym(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDsym()
        self.expr()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_expr_with_dwarf(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDwarf()
        self.expr()

    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_frame_variable_with_dsym(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDsym()
        self.frame_var()

    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_frame_variable_with_dwarf(self):
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.buildDwarf()
        self.frame_var()

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
        self.expect("expression (j->_definer->foo)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 4"])

        self.expect("expression (j->_definer->bar)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("expression *(j->_definer)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 4", "bar = 5"])

        self.expect("expression (k->foo)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 2"])

        self.expect("expression (k->bar)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 3"])

        self.expect("expression *(k)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 2", "bar = 3"])

    def frame_var(self):
        self.common_setup()

        # This should display correctly.
        self.expect("frame variable j->_definer->foo", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 4"])

        self.expect("frame variable j->_definer->bar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 5"])
            
        self.expect("frame variable *j->_definer", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 4", "bar = 5"])

        self.expect("frame variable k->foo", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 2"])

        self.expect("frame variable k->bar", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["= 3"])

        self.expect("frame variable *k", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo = 2", "bar = 3"])
                       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
