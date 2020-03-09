"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DataFormatterOneIsSingularTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_one_is_singular_with_run_command(self):
        """Test that 1 item is not as reported as 1 items."""
        self.build()
        self.oneness_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def oneness_data_formatter_commands(self):
        """Test that 1 item is not as reported as 1 items."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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

        # Now check that we are displaying Cocoa classes correctly
        self.expect('frame variable key',
                    substrs=['@"1 element"'])
        self.expect('frame variable key', matching=False,
                    substrs=['1 elements'])
        self.expect('frame variable value',
                    substrs=['@"1 element"'])
        self.expect('frame variable value', matching=False,
                    substrs=['1 elements'])
        self.expect('frame variable dict',
                    substrs=['1 key/value pair'])
        self.expect('frame variable dict', matching=False,
                    substrs=['1 key/value pairs'])
        self.expect('frame variable imset',
                    substrs=['1 index'])
        self.expect('frame variable imset', matching=False,
                    substrs=['1 indexes'])
        self.expect('frame variable binheap_ref',
                    substrs=['@"1 item"'])
        self.expect('frame variable binheap_ref', matching=False,
                    substrs=['1 items'])
        self.expect('frame variable immutableData',
                    substrs=['1 byte'])
        self.expect('frame variable immutableData', matching=False,
                    substrs=['1 bytes'])
