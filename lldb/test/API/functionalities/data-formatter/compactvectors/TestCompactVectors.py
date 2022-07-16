"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CompactVectorsFormattingTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @skipUnlessDarwin
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
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            'frame variable',
            substrs=[
                '(vFloat) valueFL = (1.25, 0, 0.25, 0)',
                '(vDouble) valueDL = (1.25, 2.25)',
                '(int16_t[8]) valueI16 = (1, 0, 4, 0, 0, 1, 0, 4)',
                '(int32_t[4]) valueI32 = (1, 0, 4, 0)',
                '(vUInt8) valueU8 = (0x01, 0x00, 0x04, 0x00, 0x00, 0x01, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)',
                '(vUInt16) valueU16 = (1, 0, 4, 0, 0, 1, 0, 4)',
                '(vUInt32) valueU32 = (1, 2, 3, 4)',
                '(vSInt8) valueS8 = (1, 0, 4, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0)',
                '(vSInt16) valueS16 = (1, 0, 4, 0, 0, 1, 0, 4)',
                '(vSInt32) valueS32 = (4, 3, 2, 1)',
                '(vBool32) valueBool32 = (0, 1, 0, 1)',
            ])
