"""Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCStepping(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    @add_test_categories(["pyapi", "basic_process"])
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr17932")
    @expectedFailureAll(oslist=["linux"], archs=no_match(["i386", "x86_64"]))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24777")
    @expectedFailureNetBSD
    def test_and_python_api(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        breakpoints_to_disable = []

        break_1_in_main = target.BreakpointCreateBySourceRegex(
            "// frame select 2, thread step-out while stopped at .c.1..",
            self.main_source_spec,
        )
        self.assertTrue(break_1_in_main, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break_1_in_main)

        break_in_a = target.BreakpointCreateBySourceRegex(
            "// break here to stop in a before calling b", self.main_source_spec
        )
        self.assertTrue(break_in_a, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break_in_a)

        break_in_b = target.BreakpointCreateBySourceRegex(
            "// thread step-out while stopped at .c.2..", self.main_source_spec
        )
        self.assertTrue(break_in_b, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break_in_b)

        break_in_c = target.BreakpointCreateBySourceRegex(
            "// Find the line number of function .c. here.", self.main_source_spec
        )
        self.assertTrue(break_in_c, VALID_BREAKPOINT)
        breakpoints_to_disable.append(break_in_c)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break_1_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        thread = threads[0]

        # Get the stop id and for fun make sure it increases:
        old_stop_id = process.GetStopID()

        # Now step over, which should cause us to hit the breakpoint in "a"
        thread.StepOver()

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break_in_a)
        if len(threads) != 1:
            self.fail("Failed to stop at breakpoint in a.")

        # Check that the stop ID increases:
        new_stop_id = process.GetStopID()
        self.assertTrue(new_stop_id > old_stop_id, "Stop ID increases monotonically.")

        thread = threads[0]

        # Step over, and we should hit the breakpoint in b:
        thread.StepOver()

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break_in_b)
        if len(threads) != 1:
            self.fail("Failed to stop at breakpoint in b.")
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

        frame.EvaluateExpression("(int) printf (print_string)")

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(
            current_line,
            frame.GetLineEntry().GetLine(),
            "The line stayed the same after expression.",
        )
        self.assertEqual(
            current_file,
            frame.GetLineEntry().GetFileSpec(),
            "The file stayed the same after expression.",
        )
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonBreakpoint,
            "We still say we stopped for a breakpoint.",
        )
        self.assertTrue(
            thread.GetStopReasonDataAtIndex(0) == current_bp[0]
            and thread.GetStopReasonDataAtIndex(1) == current_bp[1],
            "And it is the same breakpoint.",
        )

        # Also make sure running the expression didn't change the public stop id
        # but did change if we are asking for expression stops as well.
        stop_id_after_expression = process.GetStopID()
        stop_id_after_including_expressions = process.GetStopID(True)

        self.assertEqual(
            stop_id_before_expression,
            stop_id_after_expression,
            "Expression calling doesn't change stop ID",
        )

        self.assertTrue(
            stop_id_after_including_expressions > stop_id_before_including_expressions,
            "Stop ID including expressions increments over expression call.",
        )

        # Do the same thing with an expression that's going to crash, and make
        # sure we are still unchanged.

        frame.EvaluateExpression("((char *) 0)[0] = 'a'")

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(
            current_line,
            frame.GetLineEntry().GetLine(),
            "The line stayed the same after expression.",
        )
        self.assertEqual(
            current_file,
            frame.GetLineEntry().GetFileSpec(),
            "The file stayed the same after expression.",
        )
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonBreakpoint,
            "We still say we stopped for a breakpoint.",
        )
        self.assertTrue(
            thread.GetStopReasonDataAtIndex(0) == current_bp[0]
            and thread.GetStopReasonDataAtIndex(1) == current_bp[1],
            "And it is the same breakpoint.",
        )

        # Now continue and make sure we just complete the step:
        # Disable all our breakpoints first - sometimes the compiler puts two line table entries in for the
        # breakpoint a "b" and we don't want to hit that.
        for bkpt in breakpoints_to_disable:
            bkpt.SetEnabled(False)

        process.Continue()

        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "a")
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # And one more time should get us back to main:
        process.Continue()

        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # Now make sure we can call a function, break in the called function,
        # then have "continue" get us back out again:
        frame = thread.GetFrameAtIndex(0)
        frame = thread.GetFrameAtIndex(0)
        current_line = frame.GetLineEntry().GetLine()
        current_file = frame.GetLineEntry().GetFileSpec()

        break_in_b.SetEnabled(True)
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(False)
        options.SetFetchDynamicValue(False)
        options.SetUnwindOnError(False)
        frame.EvaluateExpression("b (4)", options)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, break_in_b)

        if len(threads) != 1:
            self.fail("Failed to stop at breakpoint in b when calling b.")
        thread = threads[0]

        # So do a step over here to make sure we can still do that:

        thread.StepOver()

        # See that we are still in b:
        func_name = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertEqual(func_name, "b", "Should be in 'b', were in %s" % (func_name))

        # Okay, now if we continue, we will finish off our function call and we
        # should end up back in "a" as if nothing had happened:
        process.Continue()

        self.assertEqual(
            thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), current_line
        )
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec(), current_file
        )

        # Now we are going to test step in targeting a function:

        break_in_b.SetEnabled(False)

        break_before_complex_1 = target.BreakpointCreateBySourceRegex(
            "// Stop here to try step in targeting b.", self.main_source_spec
        )
        self.assertTrue(break_before_complex_1, VALID_BREAKPOINT)

        break_before_complex_2 = target.BreakpointCreateBySourceRegex(
            "// Stop here to try step in targeting complex.", self.main_source_spec
        )
        self.assertTrue(break_before_complex_2, VALID_BREAKPOINT)

        break_before_complex_3 = target.BreakpointCreateBySourceRegex(
            "// Stop here to step targeting b and hitting breakpoint.",
            self.main_source_spec,
        )
        self.assertTrue(break_before_complex_3, VALID_BREAKPOINT)

        break_before_complex_4 = target.BreakpointCreateBySourceRegex(
            "// Stop here to make sure bogus target steps over.", self.main_source_spec
        )
        self.assertTrue(break_before_complex_4, VALID_BREAKPOINT)

        threads = lldbutil.continue_to_breakpoint(process, break_before_complex_1)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        break_before_complex_1.SetEnabled(False)

        thread.StepInto("b")
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "b")

        # Now continue out and stop at the next call to complex.  This time
        # step all the way into complex:
        threads = lldbutil.continue_to_breakpoint(process, break_before_complex_2)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        break_before_complex_2.SetEnabled(False)

        thread.StepInto("complex")
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "complex")

        # Now continue out and stop at the next call to complex.  This time
        # enable breakpoints in a and c and then step targeting b:
        threads = lldbutil.continue_to_breakpoint(process, break_before_complex_3)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        break_before_complex_3.SetEnabled(False)

        break_at_start_of_a = target.BreakpointCreateByName("a")
        break_at_start_of_c = target.BreakpointCreateByName("c")

        thread.StepInto("b")
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonBreakpoint)

        self.assertEqual(len(threads), 1)
        thread = threads[0]
        stop_break_id = thread.GetStopReasonDataAtIndex(0)
        self.assertTrue(
            stop_break_id == break_at_start_of_a.GetID()
            or stop_break_id == break_at_start_of_c.GetID()
        )

        break_at_start_of_a.SetEnabled(False)
        break_at_start_of_c.SetEnabled(False)

        process.Continue()
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "b")

        # Now continue out and stop at the next call to complex.  This time
        # enable breakpoints in a and c and then step targeting b:
        threads = lldbutil.continue_to_breakpoint(process, break_before_complex_4)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        break_before_complex_4.SetEnabled(False)

        thread.StepInto("NoSuchFunction")
        self.assertEqual(thread.GetFrameAtIndex(0).GetFunctionName(), "main")
