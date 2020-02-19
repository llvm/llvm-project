"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DataFormatterBoolRefPtr(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_boolrefptr_with_run_command(self):
        """Test the formatters we use for BOOL& and BOOL* in Objective-C."""
        self.build()
        self.boolrefptr_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.mm', '// Set break point at this line.')

    def boolrefptr_data_formatter_commands(self):
        """Test the formatters we use for BOOL& and BOOL* in Objective-C."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.mm", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        isiOS = (lldbplatformutil.getPlatform() == 'ios' or lldbplatformutil.getPlatform() == 'watchos')

        # Now check that we use the right summary for BOOL&
        self.expect('frame variable yes_ref',
                    substrs=['YES'])
        self.expect('frame variable no_ref',
                    substrs=['NO'])
        if not(isiOS):
            self.expect('frame variable unset_ref', substrs=['12'])

        # Now check that we use the right summary for BOOL*
        self.expect('frame variable yes_ptr',
                    substrs=['YES'])
        self.expect('frame variable no_ptr',
                    substrs=['NO'])
        if not(isiOS):
            self.expect('frame variable unset_ptr', substrs=['12'])

        # Now check that we use the right summary for BOOL
        self.expect('frame variable yes',
                    substrs=['YES'])
        self.expect('frame variable no',
                    substrs=['NO'])
        if not(isiOS):
            self.expect('frame variable unset', substrs=['12'])
