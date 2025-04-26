"""
Test lldb data formatter subsystem.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SkipSummaryDataFormatterTestCase(TestBase):
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24462, Data formatters have problems on Windows",
    )
    def test_with_run_command(self):
        """Test data formatter commands."""
        self.build()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        # import lldbsuite.test.lldbutil as lldbutil
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

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Setup the summaries for this scenario
        # self.runCmd("type summary add --summary-string \"${var._M_dataplus._M_p}\" std::string")
        self.runCmd('type summary add --summary-string "Level 1" "DeepData_1"')
        self.runCmd('type summary add --summary-string "Level 2" "DeepData_2" -e')
        self.runCmd('type summary add --summary-string "Level 3" "DeepData_3"')
        self.runCmd('type summary add --summary-string "Level 4" "DeepData_4"')
        self.runCmd('type summary add --summary-string "Level 5" "DeepData_5"')

        # Default case, just print out summaries
        self.expect(
            "frame variable",
            substrs=[
                "(DeepData_1) data1 = Level 1",
                "(DeepData_2) data2 = Level 2 {",
                "m_child1 = Level 3",
                "m_child2 = Level 3",
                "m_child3 = Level 3",
                "m_child4 = Level 3",
                "}",
            ],
        )

        # Skip the default (should be 1) levels of summaries
        self.expect(
            "frame variable --no-summary-depth",
            substrs=[
                "(DeepData_1) data1 = {",
                "m_child1 = 0x",
                "}",
                "(DeepData_2) data2 = {",
                "m_child1 = Level 3",
                "m_child2 = Level 3",
                "m_child3 = Level 3",
                "m_child4 = Level 3",
                "}",
            ],
        )

        # Now skip 2 levels of summaries
        self.expect(
            "frame variable --no-summary-depth=2",
            substrs=[
                "(DeepData_1) data1 = {",
                "m_child1 = 0x",
                "}",
                "(DeepData_2) data2 = {",
                "m_child1 = {",
                "m_child1 = 0x",
                "Level 4",
                "m_child2 = {",
                "m_child3 = {",
                "}",
            ],
        )

        # Check that no "Level 3" comes out
        self.expect(
            "frame variable data1.m_child1 --no-summary-depth=2",
            matching=False,
            substrs=["Level 3"],
        )

        # Now expand a pointer with 2 level of skipped summaries
        self.expect(
            "frame variable data1.m_child1 --no-summary-depth=2",
            substrs=["(DeepData_2 *) data1.m_child1 = 0x"],
        )

        # Deref and expand said pointer
        self.expect(
            "frame variable *data1.m_child1 --no-summary-depth=2",
            substrs=[
                "(DeepData_2) *data1.m_child1 = {",
                "m_child2 = {",
                "m_child1 = 0x",
                "Level 4",
                "}",
            ],
        )

        # Expand an expression, skipping 2 layers of summaries
        self.expect(
            "frame variable data1.m_child1->m_child2 --no-summary-depth=2",
            substrs=[
                "(DeepData_3) data1.m_child1->m_child2 = {",
                "m_child2 = {",
                "m_child1 = Level 5",
                "m_child2 = Level 5",
                "m_child3 = Level 5",
                "}",
            ],
        )

        # Expand same expression, skipping only 1 layer of summaries
        self.expect(
            "frame variable data1.m_child1->m_child2 --no-summary-depth=1",
            substrs=[
                "(DeepData_3) data1.m_child1->m_child2 = {",
                "m_child1 = 0x",
                "Level 4",
                "m_child2 = Level 4",
                "}",
            ],
        )

        # Expand same expression, skipping 3 layers of summaries
        self.expect(
            "frame variable data1.m_child1->m_child2 --show-types --no-summary-depth=3",
            substrs=[
                "(DeepData_3) data1.m_child1->m_child2 = {",
                'm_some_text = "Just a test"',
                "m_child2 = {",
                'm_some_text = "Just a test"',
            ],
        )

        # Change summary and expand, first without --no-summary-depth then with
        # --no-summary-depth
        self.runCmd('type summary add --summary-string "${var.m_some_text}" DeepData_5')

        self.expect(
            "fr var data2.m_child4.m_child2.m_child2",
            substrs=['(DeepData_5) data2.m_child4.m_child2.m_child2 = "Just a test"'],
        )

        self.expect(
            "fr var data2.m_child4.m_child2.m_child2 --no-summary-depth",
            substrs=[
                "(DeepData_5) data2.m_child4.m_child2.m_child2 = {",
                'm_some_text = "Just a test"',
                "}",
            ],
        )
