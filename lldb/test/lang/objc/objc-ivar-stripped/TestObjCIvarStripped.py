"""Test printing ObjC objects that use unbacked properties - so that the static ivar offsets are incorrect."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class TestObjCIvarStripped(TestBase):

    mydir = os.path.join("lang", "objc", "objc-ivar-stripped")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test that we can find stripped Objective-C ivars in the runtime"""
        self.buildDsym()
        self.objc_ivar_offsets()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.stop_line = line_number(self.main_source, '// Set breakpoint here.')

    def objc_ivar_offsets(self):
        """Test that we can find stripped Objective-C ivars in the runtime"""
        exe = os.path.join(os.getcwd(), "a.out.stripped")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.main_source, self.stop_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple (None, None, os.getcwd())
        self.assertTrue (process, "Created a process.")
        self.assertTrue (process.GetState() == lldb.eStateStopped, "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]
        
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (frame, "frame 0 is valid")
        
        # Test the expression for mc->_foo

        error = lldb.SBError()

        ivar = frame.EvaluateExpression ("(mc->_foo)")
        self.assertTrue(ivar, "Got result for mc->_foo")
        ivar_value = ivar.GetValueAsSigned (error)
        self.assertTrue (error.Success())
        self.assertTrue (ivar_value == 3)
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
