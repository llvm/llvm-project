"""Test that types defined in shared libraries work correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class SharedLibTestCase(TestBase):

    mydir = os.path.join("lang", "c", "shared_lib")

    @dsym_test
    def test_expr_with_dsym(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.buildDsym()
        self.expr()

    @dwarf_test
    def test_expr_with_dwarf(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.buildDwarf()
        self.expr()

    @dsym_test
    def test_frame_variable_with_dsym(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.buildDsym()
        self.frame_var()

    @dwarf_test
    def test_frame_variable_with_dwarf(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.buildDwarf()
        self.frame_var()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set breakpoint 0 here.')

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

    def expr(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
	self.common_setup()

        # This should display correctly.
        self.expect("expression *my_foo_ptr", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(foo)", "(sub_foo)", "other_element = 3"])

    @unittest2.expectedFailure
    # rdar://problem/10381325
    def frame_var(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
	self.common_setup()

        # This should display correctly.
        self.expect("frame variable *my_foo_ptr", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(foo)", "(sub_foo)", "other_element = 3"])
                       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
