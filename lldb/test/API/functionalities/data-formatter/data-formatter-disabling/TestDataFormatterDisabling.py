"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DataFormatterDisablingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24462, Data formatters have problems on Windows")
    def test_with_run_command(self):
        """Check that we can properly disable all data formatter categories."""
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
            self.runCmd('type category enable *', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        #self.runCmd('type category enable system VectorTypes libcxx gnu-libstdc++ CoreGraphics CoreServices AppKit CoreFoundation objc default', check=False)

        self.expect('type category list', substrs=['system', 'enabled', ])

        self.expect("frame variable numbers",
                    substrs=['[0] = 1', '[3] = 1234'])

        self.expect('frame variable string1', substrs=['hello world'])

        # now disable them all and check that nothing is formatted
        self.runCmd('type category disable *')

        self.expect("frame variable numbers", matching=False,
                    substrs=['[0] = 1', '[3] = 1234'])

        self.expect(
            'frame variable string1',
            matching=False,
            substrs=['hello world'])

        self.expect('type summary list', substrs=[
                    'Category: system (disabled)'])

        self.expect('type category list', substrs=['system', 'disabled', ])

        # now enable and check that we are back to normal
        self.runCmd("type category enable *")

        self.expect('type category list', substrs=['system', 'enabled'])

        self.expect("frame variable numbers",
                    substrs=['[0] = 1', '[3] = 1234'])

        self.expect('frame variable string1', substrs=['hello world'])

        self.expect('type category list', substrs=['system', 'enabled'])

        # last check - our cleanup will re-enable everything
        self.runCmd('type category disable *')
        self.expect('type category list', substrs=['system', 'disabled'])
