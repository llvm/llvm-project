"""
Use lldb Python API to make sure the dynamic checkers are doing their jobs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ObjCCheckerTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "objc-checker")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_objc_checker_with_dsym(self):
        """Test that checkers catch unrecognized selectors"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires Objective-C 2.0 runtime")
        self.buildDsym()
        self.do_test_checkers()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_objc_checker_with_dwarf(self):
        """Test that checkers catch unrecognized selectors"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires Objective-C 2.0 runtime")
        self.buildDwarf()
        self.do_test_checkers()

    def setUp(self):
        # Call super's setUp().                                                                                                           
        TestBase.setUp(self)

        # Find the line number to break for main.c.                                                                                       

        self.source_name = 'main.m'

    def do_test_checkers (self):
        """Make sure the dynamic checkers catch messages to unrecognized selectors"""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        
        main_bkpt = target.BreakpointCreateBySourceRegex ("Set a breakpoint here.", lldb.SBFileSpec (self.source_name))
        self.assertTrue(main_bkpt and
                        main_bkpt.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        threads = lldbutil.get_threads_stopped_at_breakpoint (process, main_bkpt)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]

        #
        #  The class Simple doesn't have a count method.  Make sure that we don't 
        #  actually try to send count but catch it as an unrecognized selector.

        frame = thread.GetFrameAtIndex(0)
        expr_value = frame.EvaluateExpression("(int) [my_simple count]", False)
        expr_error = expr_value.GetError()

        self.assertTrue (expr_error.Fail())
        
        # Make sure the call produced no NSLog stdout.
        stdout = process.GetSTDOUT(100)
        self.assertTrue (len(stdout) == 0)
        
        # Make sure the error is helpful:
        err_string = expr_error.GetCString()
        self.assertTrue ("selector" in err_string)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
