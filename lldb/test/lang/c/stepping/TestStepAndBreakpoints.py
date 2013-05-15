"""Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestObjCStepping(TestBase):

    mydir = os.path.join("lang", "c", "stepping")

    def getCategories(self):
        return ['basic_process']

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""
        self.buildDsym()
        self.step_over_stepping()

    @expectedFailureLinux # llvm.org/pr14437
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""
        self.buildDwarf()
        self.step_over_stepping()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    def step_over_stepping(self):
        """Use Python APIs to test stepping over and hitting breakpoints."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec (self.main_source)

        breakpoints_to_disable = []

        break_1_in_main = target.BreakpointCreateBySourceRegex ('// frame select 2, thread step-out while stopped at .c.1..', self.main_source_spec)
        self.assertTrue(break_1_in_main, VALID_BREAKPOINT)
        breakpoints_to_disable.append (break_1_in_main)

        break_in_a = target.BreakpointCreateBySourceRegex ('// break here to stop in a before calling b', self.main_source_spec)
        self.assertTrue(break_in_a, VALID_BREAKPOINT)
        breakpoints_to_disable.append (break_in_a)

        break_in_b = target.BreakpointCreateBySourceRegex ('// thread step-out while stopped at .c.2..', self.main_source_spec)
        self.assertTrue(break_in_b, VALID_BREAKPOINT)
        breakpoints_to_disable.append (break_in_b)

        break_in_c = target.BreakpointCreateBySourceRegex ('// Find the line number of function .c. here.', self.main_source_spec)
        self.assertTrue(break_in_c, VALID_BREAKPOINT)
        breakpoints_to_disable.append (break_in_c)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break_1_in_main)

        if len(threads) != 1:
            self.fail ("Failed to stop at first breakpoint in main.")

        thread = threads[0]

        # Get the stop id and for fun make sure it increases:
        old_stop_id = process.GetStopID()

        # Now step over, which should cause us to hit the breakpoint in "a"
        thread.StepOver()

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break_in_a)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint in a.")

        # Check that the stop ID increases:
        new_stop_id = process.GetStopID()
        self.assertTrue(new_stop_id > old_stop_id, "Stop ID increases monotonically.")

        thread = threads[0]

        # Step over, and we should hit the breakpoint in b:
        thread.StepOver()

        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break_in_b)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint in b.")
        thread = threads[0]

        # Now try running some function, and make sure that we still end up in the same place
        # and with the same stop reason.
        frame = thread.GetFrameAtIndex(0)
        current_line = frame.GetLineEntry().GetLine()
        current_file = frame.GetLineEntry().GetFileSpec()
        current_bp = []
        current_bp.append(thread.GetStopReasonDataAtIndex(0))
        current_bp.append(thread.GetStopReasonDataAtIndex(1))

        stop_id_before_expression = process.GetStopID()
        stop_id_before_including_expressions = process.GetStopID(True)

        frame.EvaluateExpression ("(int) printf (print_string)")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (current_line == frame.GetLineEntry().GetLine(), "The line stayed the same after expression.")
        self.assertTrue (current_file == frame.GetLineEntry().GetFileSpec(), "The file stayed the same after expression.")
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonBreakpoint, "We still say we stopped for a breakpoint.")
        self.assertTrue (thread.GetStopReasonDataAtIndex(0) == current_bp[0] and thread.GetStopReasonDataAtIndex(1) == current_bp[1], "And it is the same breakpoint.")
        
        # Also make sure running the expression didn't change the public stop id
        # but did change if we are asking for expression stops as well.
        stop_id_after_expression = process.GetStopID()
        stop_id_after_including_expressions = process.GetStopID(True)

        self.assertTrue (stop_id_before_expression == stop_id_after_expression, "Expression calling doesn't change stop ID")

        self.assertTrue (stop_id_after_including_expressions > stop_id_before_including_expressions, "Stop ID including expressions increments over expression call.")

        # Do the same thing with an expression that's going to crash, and make sure we are still unchanged.

        frame.EvaluateExpression ("((char *) 0)[0] = 'a'")

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (current_line == frame.GetLineEntry().GetLine(), "The line stayed the same after expression.")
        self.assertTrue (current_file == frame.GetLineEntry().GetFileSpec(), "The file stayed the same after expression.")
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonBreakpoint, "We still say we stopped for a breakpoint.")
        self.assertTrue (thread.GetStopReasonDataAtIndex(0) == current_bp[0] and thread.GetStopReasonDataAtIndex(1) == current_bp[1], "And it is the same breakpoint.")

        # Now continue and make sure we just complete the step:
        # Disable all our breakpoints first - sometimes the compiler puts two line table entries in for the
        # breakpoint a "b" and we don't want to hit that.
        for bkpt in breakpoints_to_disable:
            bkpt.SetEnabled(False)

        process.Continue()

        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "a")
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        # And one more time should get us back to main:
        process.Continue()

        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "main")
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        # Now make sure we can call a function, break in the called function, then have "continue" get us back out again:
        frame = thread.GetFrameAtIndex(0)
        frame = thread.GetFrameAtIndex(0)
        current_line = frame.GetLineEntry().GetLine()
        current_file = frame.GetLineEntry().GetFileSpec()

        break_in_b.SetEnabled(True)
        frame.EvaluateExpression ("b (4)", lldb.eNoDynamicValues, False)

        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break_in_b)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint in b when calling b.")
        thread = threads[0]

        # So do a step over here to make sure we can still do that:

        thread.StepOver()

        # See that we are still in b:
        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "b")

        # Okay, now if we continue, we will finish off our function call and we should end up back in "a" as if nothing had happened:
        process.Continue ()

        self.assertTrue (thread.GetFrameAtIndex(0).GetLineEntry().GetLine() == current_line)
        self.assertTrue (thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec() == current_file)

        # Now we are going to test step in targetting a function:

        break_in_b.SetEnabled (False)

        break_before_complex_1 = target.BreakpointCreateBySourceRegex ('// Stop here to try step in targetting b.', self.main_source_spec)
        self.assertTrue(break_before_complex_1, VALID_BREAKPOINT)

        break_before_complex_2 = target.BreakpointCreateBySourceRegex ('// Stop here to try step in targetting complex.', self.main_source_spec)
        self.assertTrue(break_before_complex_2, VALID_BREAKPOINT)

        break_before_complex_3 = target.BreakpointCreateBySourceRegex ('// Stop here to step targetting b and hitting breakpoint.', self.main_source_spec)
        self.assertTrue(break_before_complex_3, VALID_BREAKPOINT)

        break_before_complex_4 = target.BreakpointCreateBySourceRegex ('// Stop here to make sure bogus target steps over.', self.main_source_spec)
        self.assertTrue(break_before_complex_4, VALID_BREAKPOINT)

        threads = lldbutil.continue_to_breakpoint(process, break_before_complex_1)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        break_before_complex_1.SetEnabled(False)

        thread.StepInto ("b")
        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "b")

        # Now continue out and stop at the next call to complex.  This time step all the way into complex:
        threads = lldbutil.continue_to_breakpoint (process, break_before_complex_2)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        break_before_complex_2.SetEnabled(False)

        thread.StepInto ("complex")
        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "complex")
        
        # Now continue out and stop at the next call to complex.  This time enable breakpoints in a and c and then step targetting b:
        threads = lldbutil.continue_to_breakpoint (process, break_before_complex_3)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        break_before_complex_3.SetEnabled(False)

        break_at_start_of_a = target.BreakpointCreateByName ('a')
        break_at_start_of_c = target.BreakpointCreateByName ('c')

        thread.StepInto ("b")
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonBreakpoint);

        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        stop_break_id = thread.GetStopReasonDataAtIndex(0)
        self.assertTrue(stop_break_id == break_at_start_of_a.GetID() or stop_break_id == break_at_start_of_c.GetID())

        break_at_start_of_a.SetEnabled(False)
        break_at_start_of_c.SetEnabled(False)

        process.Continue()
        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "b")
        
        # Now continue out and stop at the next call to complex.  This time enable breakpoints in a and c and then step targetting b:
        threads = lldbutil.continue_to_breakpoint (process, break_before_complex_4)
        self.assertTrue (len(threads) == 1)
        thread = threads[0]
        break_before_complex_4.SetEnabled(False)

        thread.StepInto("NoSuchFunction")
        self.assertTrue (thread.GetFrameAtIndex(0).GetFunctionName() == "main")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
