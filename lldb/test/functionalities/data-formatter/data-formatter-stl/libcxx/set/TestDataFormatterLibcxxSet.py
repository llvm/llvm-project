"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LibcxxSetDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-stl", "libcxx", "set")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    @skipIfLinux # No standard locations for libc++ on Linux, so skip for now 
    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp (self, "Set break point at this line.")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect('image list',substrs=['libc++.1.dylib','libc++abi.dylib'])

        self.expect("frame variable ii",substrs = ["size=0","{}"])
        self.runCmd("continue")
        self.expect("frame variable ii",substrs = ["size=6","[0] = 0","[1] = 1", "[2] = 2", "[3] = 3", "[4] = 4", "[5] = 5"])
        self.runCmd("continue")
        self.expect("frame variable ii",substrs = ["size=7","[2] = 2", "[3] = 3", "[6] = 6"])
        self.expect("frame variable ii[2]",substrs = [" = 2"])
        self.expect("p ii",substrs = ["size=7","[2] = 2", "[3] = 3", "[6] = 6"])
        self.runCmd("continue")
        self.expect("frame variable ii",substrs = ["size=0","{}"])
        self.runCmd("continue")
        self.expect("frame variable ii",substrs = ["size=0","{}"])
        self.expect("frame variable ss",substrs = ["size=0","{}"])
        self.runCmd("continue")
        self.expect("frame variable ss",substrs = ["size=2",'[0] = "a"','[1] = "a very long string is right here"'])
        self.runCmd("continue")
        self.expect("frame variable ss",substrs = ["size=4",'[2] = "b"','[3] = "c"','[0] = "a"','[1] = "a very long string is right here"'])
        self.expect("p ss",substrs = ["size=4",'[2] = "b"','[3] = "c"','[0] = "a"','[1] = "a very long string is right here"'])
        self.expect("frame variable ss[2]",substrs = [' = "b"'])
        self.runCmd("continue")
        self.expect("frame variable ss",substrs = ["size=3",'[0] = "a"','[1] = "a very long string is right here"','[2] = "c"'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
