"""Test that forward declaration of a data structure gets resolved correctly."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ForwardDeclarationTestCase(TestBase):

    mydir = os.path.join("lang", "c", "forward")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.buildDsym()
        self.forward_declaration()

    # rdar://problem/8648070
    # 'expression *bar_ptr' seg faults
    # rdar://problem/8546815
    # './dotest.py -v -t forward' fails for test_with_dwarf_and_run_command
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.buildDwarf()
        self.forward_declaration()

    def forward_declaration(self):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_symbol (self, "foo", num_expected_locations=1, sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # This should display correctly.
        # Note that the member fields of a = 1 and b = 2 is by design.
        self.expect("frame variable --show-types *bar_ptr", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(bar) *bar_ptr = ',
                       '(int) a = 1',
                       '(int) b = 2'])

        # And so should this.
        self.expect("expression --show-types -- *bar_ptr", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(bar)',
                       '(int) a = 1',
                       '(int) b = 2'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
