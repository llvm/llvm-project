"""Test that types defined in shared libraries work correctly."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRealDefinition(TestBase):

    def test_frame_var_after_stop_at_interface(self):
        """Test that we can find the implementation for an objective C type"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()

        lldbutil.run_to_source_breakpoint(
            self,
            '// Set breakpoint where Bar is an interface',
            lldb.SBFileSpec("Foo.m", False))

        # Break inside the foo function which takes a bar_ptr argument.
        self.expect('breakpoint set -p "// Set breakpoint in main"')
        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        # This should display correctly.
        self.expect(
            "frame variable foo->_bar->_hidden_ivar",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(NSString *)",
                "foo->_bar->_hidden_ivar = 0x"])

    def test_frame_var_after_stop_at_implementation(self):
        """Test that we can find the implementation for an objective C type"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()

        lldbutil.run_to_source_breakpoint(
            self,
            '// Set breakpoint where Bar is an implementation',
            lldb.SBFileSpec("Bar.m", False))

        self.expect('breakpoint set -p "// Set breakpoint in main"')
        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        # This should display correctly.
        self.expect(
            "frame variable foo->_bar->_hidden_ivar",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(NSString *)",
                "foo->_bar->_hidden_ivar = 0x"])
