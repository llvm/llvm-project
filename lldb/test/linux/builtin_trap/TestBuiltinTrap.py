"""
Test lldb ability to unwind a stack with a function containing a call to the
'__builtin_trap' intrinsic, which GCC (4.6) encodes to an illegal opcode.
"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil

class BuiltinTrapTestCase(TestBase):

    mydir = os.path.join("linux", "builtin_trap")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test that LLDB handles a function with __builtin_trap correctly."""
        self.buildDsym()
        self.builtin_trap_unwind()

    @dwarf_test
    @expectedFailureGcc # llvm.org/pr15936: LLDB is omits a function that contains an
                        #           illegal opcode from backtraces. This
                        #           failure is GCC 4.6 specific.
    def test_with_dwarf_and_run_command(self):
        """Test that LLDB handles a function with __builtin_trap correctly."""
        self.buildDwarf()
        self.builtin_trap_unwind()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def builtin_trap_unwind(self):
        """Test that LLDB handles unwinding a frame that contains a function
           with a __builtin_trap intrinsic.
        """
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line,
                                                 num_expected_locations=1,
                                                 loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # print backtrace, expect both 'bar' and 'main' functions to be listed
        self.expect('bt', substrs = ['bar', 'main'])

        # go up one frame
        self.runCmd("up", RUN_SUCCEEDED)

        # evaluate a local
        self.expect('p foo', substrs = ['= 5'])



if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
