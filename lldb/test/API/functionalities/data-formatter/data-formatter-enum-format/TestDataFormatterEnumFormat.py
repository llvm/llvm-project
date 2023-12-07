"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class EnumFormatTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.expect(
            "frame variable",
            substrs=["(Foo) f = Case45", "(int) x = 1", "(int) y = 45", "(int) z = 43"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type format add --type Foo int")

        # The type format list should show our custom formats.
        self.expect("type format list -w default", substrs=["int: as type Foo"])

        self.expect(
            "frame variable",
            substrs=[
                "(Foo) f = Case45",
                "(int) x = Case1",
                "(int) y = Case45",
                "(int) z = 43",
            ],
        )
