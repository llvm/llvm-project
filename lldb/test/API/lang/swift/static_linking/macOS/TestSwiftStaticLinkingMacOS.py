"""
Test that macOS can statically link two separately-compiled Swift modules
with one Objective-C module, link them through the clang driver, and still
access debug info for each of the Swift modules.
"""
# System imports
import os

# Third-party imports

# LLDB imports
import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbtest, lldbutil


class SwiftStaticLinkingMacOSTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def expect_self_var_available_at_breakpoint(
            self, process, breakpoint, module_name):
        patterns = [
            # Ensure we report a self with an address.
            r"=\s*0x[0-9a-fA-F]+",
            # Ensure we think it is an NSObject.
            r"ObjectiveC.NSObject"]
        substrs = [
            "(%s.%s)" % (module_name, module_name)
        ]
        self.expect("expr self", patterns=patterns,
                    substrs=substrs)

    @skipUnlessDarwin
    @swiftTest
    def test_variables_print_from_both_swift_modules(self):
        """Test that variables from two modules can be accessed."""
        self.build()

        # Create the target
        target, process, _, breakpoint_a = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("A.swift"), 5)

        # breakpoint_b = target.BreakpointCreateBySourceRegex(
        #     'Set breakpoint here', src_b)
        breakpoint_b = target.BreakpointCreateByLocation(
            lldb.SBFileSpec("B.swift"), 5)
        self.assertTrue(breakpoint_b.GetNumLocations() > 0,
                        lldbtest.VALID_BREAKPOINT)

        # We should be at breakpoint in module A.
        self.expect_self_var_available_at_breakpoint(
            process, breakpoint_a, "A")

        # Jump to the next breakpoint
        process.Continue()

        # We should be at breakpoint in module B.
        self.expect_self_var_available_at_breakpoint(
            process, breakpoint_b, "B")

        return
