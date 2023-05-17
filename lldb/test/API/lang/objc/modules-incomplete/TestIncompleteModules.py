"""Test that DWARF types are trusted over module types"""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class IncompleteModulesTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    @add_test_categories(["gmodules"])
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        self.runCmd(
            "settings set target.clang-module-search-paths \"" +
            self.getSourceDir() +
            "\"")

        self.expect("expr @import myModule; 3", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["int", "3"])

        self.expect(
            "expr private_func()",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "int",
                "5"])

        self.expect("expr MY_MIN(2,3)", "#defined macro was found",
                    substrs=["int", "2"])

        self.expect("expr MY_MAX(2,3)", "#undefd macro was correctly not found",
                    error=True)
