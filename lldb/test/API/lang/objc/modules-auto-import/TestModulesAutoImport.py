"""Test that importing modules in Objective-C works as expected."""



import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCModulesAutoImportTestCase(TestBase):

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

        self.runCmd("settings set target.auto-import-clang-modules true")

        self.expect("expression getpid()", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["pid_t"])
