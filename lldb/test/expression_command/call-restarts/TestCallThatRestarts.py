"""
Test calling a function that hits a signal set to auto-restart, make sure the call completes.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class ExprCommandWithTimeoutsTestCase(TestBase):

    mydir = os.path.join("expression_command", "call-restarts")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "lotta-signals.c"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test calling std::String member function."""
        self.buildDsym()
        self.call_function()

    @skipIfLinux # llvm.org/pr15278: handle expressions that generate signals on Linux
    @dwarf_test
    def test_with_dwarf(self):
        """Test calling std::String member function."""
        self.buildDwarf()
        self.call_function()

    def check_after_call (self, num_sigchld):
        after_call = self.sigchld_no.GetValueAsSigned(-1)
        self.assertTrue (after_call - self.start_sigchld_no == num_sigchld, "Really got %d SIGCHLD signals through the call."%(num_sigchld))
        self.start_sigchld_no = after_call

        # Check that we are back where we were before:
        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue (self.orig_frame_pc == frame.GetPC(), "Restored the zeroth frame correctly")

        
    def call_function(self):
        """Test calling function with timeout."""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        empty = lldb.SBFileSpec()
        breakpoint = target.BreakpointCreateBySourceRegex('Stop here in main.',self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        
        # Make sure the SIGCHLD behavior is pass/no-stop/no-notify:
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("process handle SIGCHLD -s 0 -p 1 -n 0", return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Set SIGCHLD to pass, no-stop")

        # The sigchld_no variable should be 0 at this point.
        self.sigchld_no = target.FindFirstGlobalVariable("sigchld_no")
        self.assertTrue (self.sigchld_no.IsValid(), "Got a value for sigchld_no")

        self.start_sigchld_no = self.sigchld_no.GetValueAsSigned (-1)
        self.assertTrue (self.start_sigchld_no != -1, "Got an actual value for sigchld_no")

        options = lldb.SBExpressionOptions()
        options.SetUnwindOnError(True)

        frame = self.thread.GetFrameAtIndex(0)
        # Store away the PC to check that the functions unwind to the right place after calls
        self.orig_frame_pc = frame.GetPC()

        num_sigchld = 30
        value = frame.EvaluateExpression ("call_me (%d)"%(num_sigchld), options)
        self.assertTrue (value.IsValid())
        self.assertTrue (value.GetError().Success() == True)
        self.assertTrue (value.GetValueAsSigned(-1) == num_sigchld)

        self.check_after_call(num_sigchld)

        # Okay, now try with a breakpoint in the called code in the case where
        # we are ignoring breakpoint hits.
        handler_bkpt = target.BreakpointCreateBySourceRegex("Got sigchld %d.", self.main_source_spec)
        self.assertTrue (handler_bkpt.GetNumLocations() > 0)
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression("call_me (%d)"%(num_sigchld), options)

        self.assertTrue (value.IsValid() and value.GetError().Success() == True)
        self.assertTrue (value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Now set the signal to print but not stop and make sure that calling still works:
        self.dbg.GetCommandInterpreter().HandleCommand("process handle SIGCHLD -s 0 -p 1 -n 1", return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Set SIGCHLD to pass, no-stop, notify")

        value = frame.EvaluateExpression("call_me (%d)"%(num_sigchld), options)

        self.assertTrue (value.IsValid() and value.GetError().Success() == True)
        self.assertTrue (value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Now set this unwind on error to false, and make sure that we still complete the call:
        options.SetUnwindOnError(False)
        value = frame.EvaluateExpression("call_me (%d)"%(num_sigchld), options)

        self.assertTrue (value.IsValid() and value.GetError().Success() == True)
        self.assertTrue (value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Okay, now set UnwindOnError to true, and then make the signal behavior to stop
        # and see that now we do stop at the signal point:
        
        self.dbg.GetCommandInterpreter().HandleCommand("process handle SIGCHLD -s 1 -p 1 -n 1", return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Set SIGCHLD to pass, stop, notify")
        
        value = frame.EvaluateExpression("call_me (%d)"%(num_sigchld), options)
        self.assertTrue (value.IsValid() and value.GetError().Success() == False)
        
        # Set signal handling back to no-stop, and continue and we should end up back in out starting frame:
        self.dbg.GetCommandInterpreter().HandleCommand("process handle SIGCHLD -s 0 -p 1 -n 1", return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Set SIGCHLD to pass, no-stop, notify")

        error = process.Continue()
        self.assertTrue (error.Success(), "Continuing after stopping for signal succeeds.")
        
        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue (frame.GetPC() == self.orig_frame_pc, "Continuing returned to the place we started.")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
