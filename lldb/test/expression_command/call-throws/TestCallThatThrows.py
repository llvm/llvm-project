"""
Test calling a function that throws an ObjC exception, make sure that it doesn't propagate the exception.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandWithThrowTestCase(TestBase):

    mydir = os.path.join("expression_command", "call-throws")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "call-throws.m"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test calling a function that throws and ObjC exception."""
        self.buildDsym()
        self.call_function()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin due to ObjC test case")
    @dwarf_test
    def test_with_dwarf(self):
        """Test calling a function that throws and ObjC exception."""
        self.buildDwarf()
        self.call_function()

    def check_after_call (self):
        # Check that we are back where we were before:
        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue (self.orig_frame_pc == frame.GetPC(), "Restored the zeroth frame correctly")

        
    def call_function(self):
        """Test calling function that throws."""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex('I am about to throw.',self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        
        options = lldb.SBExpressionOptions()
        options.SetUnwindOnError(True)

        frame = self.thread.GetFrameAtIndex(0)
        # Store away the PC to check that the functions unwind to the right place after calls
        self.orig_frame_pc = frame.GetPC()

        value = frame.EvaluateExpression ("[my_class callMeIThrow]", options)
        self.assertTrue (value.IsValid())
        self.assertTrue (value.GetError().Success() == False)

        self.check_after_call()

        # Okay, now try with a breakpoint in the called code in the case where
        # we are ignoring breakpoint hits.
        handler_bkpt = target.BreakpointCreateBySourceRegex("I felt like it", self.main_source_spec)
        self.assertTrue (handler_bkpt.GetNumLocations() > 0)
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression ("[my_class callMeIThrow]", options)

        self.assertTrue (value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now set the ObjC language breakpoint and make sure that doesn't interfere with the call:
        exception_bkpt = target.BreakpointCreateForException (lldb.eLanguageTypeObjC, False, True)
        self.assertTrue(exception_bkpt.GetNumLocations() > 0)

        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression ("[my_class callMeIThrow]", options)

        self.assertTrue (value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now set this unwind on error to false, and make sure that we stop where the exception was thrown
        options.SetUnwindOnError(False)
        value = frame.EvaluateExpression ("[my_class callMeIThrow]", options)


        self.assertTrue (value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
