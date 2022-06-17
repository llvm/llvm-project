"""
Test that the user can input a format but it will not prevail over summary format's choices.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class UserFormatVSSummaryTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that the user can input a format but it will not prevail over summary format's choices."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("frame variable p1", substrs=[
                    '(Pair) p1 = (x = 3, y = -3)'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd('type summary add Pair -s "x=${var.x%d},y=${var.y%u}"')

        self.expect("frame variable p1", substrs=[
                    '(Pair) p1 = x=3,y=4294967293'])
        self.expect(
            "frame variable -f x p1",
            substrs=['(Pair) p1 = x=0x00000003,y=0xfffffffd'],
            matching=False)
        self.expect(
            "frame variable -f d p1",
            substrs=['(Pair) p1 = x=3,y=-3'],
            matching=False)
        self.expect("frame variable p1", substrs=[
                    '(Pair) p1 = x=3,y=4294967293'])

        self.runCmd('type summary add Pair -s "x=${var.x%x},y=${var.y%u}"')

        self.expect("frame variable p1", substrs=[
                    '(Pair) p1 = x=0x00000003,y=4294967293'])
        self.expect(
            "frame variable -f d p1",
            substrs=['(Pair) p1 = x=3,y=-3'],
            matching=False)
