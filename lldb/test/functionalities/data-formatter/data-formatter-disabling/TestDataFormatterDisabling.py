"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class DataFormatterDisablingTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-disabling")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Check that we can properly disable all data formatter categories."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
             self.runCmd('type category enable default', check=False)
             self.runCmd('type category enable system', check=False)
             self.runCmd('type category enable VectorTypes', check=False)
             self.runCmd('type category enable libcxx', check=False)
             self.runCmd('type category enable gnu-libstdc++', check=False)
             self.runCmd('type category enable CoreGraphics', check=False)
             self.runCmd('type category enable CoreServices', check=False)
             self.runCmd('type category enable AppKit', check=False)
             self.runCmd('type category enable CoreFoundation', check=False)
             self.runCmd('type category enable objc', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        #self.runCmd('type category enable system VectorTypes libcxx gnu-libstdc++ CoreGraphics CoreServices AppKit CoreFoundation objc default', check=False)

        self.expect('type category list', substrs = ['system is enabled', 'gnu-libstdc++ is enabled', 'AppKit is enabled'])

        self.expect("frame variable numbers",
            substrs = ['[0] = 1', '[3] = 1234'])

        self.expect('frame variable string1', substrs = ['hello world'])

        # now disable them all and check that nothing is formatted
        self.runCmd('type category disable *')

        self.expect("frame variable numbers", matching=False,
            substrs = ['[0] = 1', '[3] = 1234'])

        self.expect('frame variable string1', matching=False, substrs = ['hello world'])

        self.expect('type category list', substrs = ['system is not enabled', 'gnu-libstdc++ is not enabled', 'AppKit is not enabled'])
        
        # now enable and check that we are back to normal
        self.runCmd("type category enable *")

        self.expect('type category list', substrs = ['system is enabled', 'gnu-libstdc++ is enabled', 'AppKit is enabled'])

        self.expect("frame variable numbers",
            substrs = ['[0] = 1', '[3] = 1234'])

        self.expect('frame variable string1', substrs = ['hello world'])

        self.expect('type category list', substrs = ['system is enabled', 'gnu-libstdc++ is enabled', 'AppKit is enabled'])

        # last check - our cleanup will re-enable everything
        self.runCmd('type category disable *')
        self.expect('type category list', substrs = ['system is not enabled', 'gnu-libstdc++ is not enabled', 'AppKit is not enabled'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
