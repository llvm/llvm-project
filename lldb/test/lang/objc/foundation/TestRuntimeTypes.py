"""
Test that Objective-C methods from the runtime work correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class RuntimeTypesTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")

    @dsym_test
    def test_break_with_dsym(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        # This only applies to the v2 runtime
        if self.getArchitecture() == 'x86_64':
            self.buildDsym()
            self.runtime_types()

    @dwarf_test
    def test_break_with_dwarf(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        # This only applies to the v2 runtime
        if self.getArchitecture() == 'x86_64':
            self.buildDwarf()
            self.runtime_types()

    def runtime_types(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at -[MyString description].
        lldbutil.run_break_set_by_symbol (self, '-[MyString description]', num_expected_locations=1, sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The backtrace should show we stop at -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        # Use runtime information about NSString.

        # The length property should be usable.
        self.expect("expression str.length", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(unsigned long long)"])

        # Static methods on NSString should work.
        self.expect("expr [NSString stringWithCString:\"foo\" encoding:1]", VALID_TYPE,
            substrs = ["(id)", "$1"])

        self.expect("po $1", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["foo"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
