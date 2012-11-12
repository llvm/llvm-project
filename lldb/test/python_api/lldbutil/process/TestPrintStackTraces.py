"""
Test SBprocess and SBThread APIs with printing of the stack traces using lldbutil.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class ThreadsStackTracesTestCase(TestBase):

    mydir = "python_api/lldbutil/process"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @python_api_test
    def test_stack_traces(self):
        """Test SBprocess and SBThread APIs with printing of the stack traces."""
        self.buildDefault()
        self.break_and_print_stacktraces()

    def break_and_print_stacktraces(self):
        """Break at main.cpp:68 and do a threads dump"""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(["abc", "xyz"], None, os.getcwd())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        import lldbutil
        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
            substrs = ['(int)argc=3'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
