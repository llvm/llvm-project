"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class DataFormatterHexCapsTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format delete hex', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type format add -f uppercase int")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             'first = 0x001122AA', 'second = 0x1122BB44'])

        self.runCmd("type format add -f hex int")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             'first = 0x001122aa', 'second = 0x1122bb44'])

        self.runCmd("type format delete int")

        self.runCmd(
            "type summary add -s \"${var.first%X} and ${var.second%x}\" foo")

        self.expect('frame variable mine',
                    substrs=['(foo) mine = 0x001122AA and 0x1122bb44'])

        self.runCmd(
            "type summary add -s \"${var.first%X} and ${var.second%X}\" foo")
        self.runCmd("next")
        self.runCmd("next")
        self.expect('frame variable mine',
                    substrs=['(foo) mine = 0xAABBCCDD and 0x1122BB44'])

        self.runCmd(
            "type summary add -s \"${var.first%x} and ${var.second%X}\" foo")
        self.expect('frame variable mine',
                    substrs=['(foo) mine = 0xaabbccdd and 0x1122BB44'])
        self.runCmd("next")
        self.runCmd("next")
        self.runCmd(
            "type summary add -s \"${var.first%x} and ${var.second%x}\" foo")
        self.expect('frame variable mine',
                    substrs=['(foo) mine = 0xaabbccdd and 0xff00ff00'])
        self.runCmd(
            "type summary add -s \"${var.first%X} and ${var.second%X}\" foo")
        self.expect('frame variable mine',
                    substrs=['(foo) mine = 0xAABBCCDD and 0xFF00FF00'])
