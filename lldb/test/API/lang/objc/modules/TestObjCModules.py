"""Test that importing modules in Objective-C works as expected."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCModulesTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    @skipIf(macos_version=["<", "10.12"])
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.expect("expr @import Darwin; 3", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["int", "3"])

        self.expect("expr getpid()", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["pid_t"])

        self.expect(
            "expr @import Foundation; 4",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "int",
                "4"])

        # Type lookup should still work and print something reasonable
        # for types from the module.
        self.expect("type lookup NSObject", substrs=["instanceMethod"])

        self.expect("expr string.length", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["NSUInteger", "5"])

        self.expect("expr array.count", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["NSUInteger", "3"])

        self.expect(
            "p *[NSURL URLWithString:@\"http://lldb.llvm.org\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSURL",
                "isa",
                "_urlString"])

        self.expect(
            "p [NSURL URLWithString:@\"http://lldb.llvm.org\"].scheme",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["http"])
        # Test that the NULL macro still works with a loaded module.
        self.expect_expr("int *i = NULL; i == NULL", result_value="true")
