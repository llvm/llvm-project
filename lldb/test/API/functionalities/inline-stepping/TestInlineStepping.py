"""Test stepping over and into inlined functions."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlineStepping(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @expectedFailureAll(
        compiler="icc",
        bugnumber="# Not really a bug.  ICC combines two inlined functions.")
    def test_with_python_api(self):
        """Test stepping over and into inlined functions."""
        self.build()
        self.inline_stepping()

    @add_test_categories(['pyapi'])
    def test_step_over_with_python_api(self):
        """Test stepping over and into inlined functions."""
        self.build()
        self.inline_stepping_step_over()

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_step_in_template_with_python_api(self):
        """Test stepping in to templated functions."""
        self.build()
        self.step_in_template()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "calling.cpp"
        self.source_lines = {}
        functions = [
            'caller_ref_1',
            'caller_ref_2',
            'inline_ref_1',
            'inline_ref_2',
            'called_by_inline_ref',
            'caller_trivial_1',
            'caller_trivial_2',
            'inline_trivial_1',
            'inline_trivial_2',
            'called_by_inline_trivial']
        for name in functions:
            self.source_lines[name] = line_number(
                self.main_source, "// In " + name + ".")
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_step(self, step_type, destination_line_entry, test_stack_depth):
        expected_stack_depth = self.thread.GetNumFrames()
        if step_type == "into":
            expected_stack_depth += 1
            self.thread.StepInto()
        elif step_type == "out":
            expected_stack_depth -= 1
            self.thread.StepOut()
        elif step_type == "over":
            self.thread.StepOver()
        else:
            self.fail("Unrecognized step type: " + step_type)

        threads = lldbutil.get_stopped_threads(
            self.process, lldb.eStopReasonPlanComplete)
        if len(threads) != 1:
            destination_description = lldb.SBStream()
            destination_line_entry.GetDescription(destination_description)
            self.fail(
                "Failed to stop due to step " +
                step_type +
                " operation stepping to: " +
                destination_description.GetData())

        self.thread = threads[0]

        stop_line_entry = self.thread.GetFrameAtIndex(0).GetLineEntry()
        self.assertTrue(
            stop_line_entry.IsValid(),
            "Stop line entry was not valid.")

        # Don't use the line entry equal operator because we don't care about
        # the column number.
        stop_at_right_place = (stop_line_entry.GetFileSpec() == destination_line_entry.GetFileSpec(
        ) and stop_line_entry.GetLine() == destination_line_entry.GetLine())
        if not stop_at_right_place:
            destination_description = lldb.SBStream()
            destination_line_entry.GetDescription(destination_description)

            actual_description = lldb.SBStream()
            stop_line_entry.GetDescription(actual_description)

            self.fail(
                "Step " +
                step_type +
                " stopped at wrong place: expected: " +
                destination_description.GetData() +
                " got: " +
                actual_description.GetData() +
                ".")

        real_stack_depth = self.thread.GetNumFrames()

        if test_stack_depth and real_stack_depth != expected_stack_depth:
            destination_description = lldb.SBStream()
            destination_line_entry.GetDescription(destination_description)
            self.fail(
                "Step %s to %s got wrong number of frames, should be: %d was: %d." %
                (step_type,
                 destination_description.GetData(),
                 expected_stack_depth,
                 real_stack_depth))

    def run_step_sequence(self, step_sequence):
        """This function takes a list of duples instructing how to run the program.  The first element in each duple is
           a source pattern for the target location, and the second is the operation that will take you from the current
           source location to the target location.  It will then run all the steps in the sequence.
           It will check that you arrived at the expected source location at each step, and that the stack depth changed
           correctly for the operation in the sequence."""

        target_line_entry = lldb.SBLineEntry()
        target_line_entry.SetFileSpec(self.main_source_spec)

        test_stack_depth = True
        # Work around for <rdar://problem/16363195>, the darwin unwinder seems flakey about whether it duplicates the first frame
        # or not, which makes counting stack depth unreliable.
        if self.platformIsDarwin():
            test_stack_depth = False

        for step_pattern in step_sequence:
            step_stop_line = line_number(self.main_source, step_pattern[0])
            target_line_entry.SetLine(step_stop_line)
            self.do_step(step_pattern[1], target_line_entry, test_stack_depth)

    def inline_stepping(self):
        """Use Python APIs to test stepping over and hitting breakpoints."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        break_1_in_main = target.BreakpointCreateBySourceRegex(
            '// Stop here and step over to set up stepping over.', self.main_source_spec)
        self.assertTrue(break_1_in_main, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, break_1_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        self.thread = threads[0]

        # Step over the inline_value = 0 line to get us to inline_trivial_1 called from main.  Doing it this way works
        # around a bug in lldb where the breakpoint on the containing line of an inlined function with no return value
        # gets set past the insertion line in the function.
        # Then test stepping over a simple inlined function.  Note, to test all the parts of the inlined stepping
        # the calls inline_stepping_1 and inline_stepping_2 should line up at the same address, that way we will test
        # the "virtual" stepping.
        # FIXME: Put in a check to see if that is true and warn if it is not.

        step_sequence = [["// At inline_trivial_1 called from main.", "over"],
                         ["// At first call of caller_trivial_1 in main.", "over"]]
        self.run_step_sequence(step_sequence)

        # Now step from caller_ref_1 all the way into called_by_inline_trivial

        step_sequence = [["// In caller_trivial_1.", "into"],
                         ["// In caller_trivial_2.", "into"],
                         ["// In inline_trivial_1.", "into"],
                         ["// In inline_trivial_2.", "into"],
                         ["// At caller_by_inline_trivial in inline_trivial_2.", "over"],
                         ["// In called_by_inline_trivial.", "into"]]
        self.run_step_sequence(step_sequence)

        # Now run to the inline_trivial_1 just before the immediate step into
        # inline_trivial_2:

        break_2_in_main = target.BreakpointCreateBySourceRegex(
            '// At second call of caller_trivial_1 in main.', self.main_source_spec)
        self.assertTrue(break_2_in_main, VALID_BREAKPOINT)

        threads = lldbutil.continue_to_breakpoint(
            self.process, break_2_in_main)
        self.assertEqual(
            len(threads), 1,
            "Successfully ran to call site of second caller_trivial_1 call.")
        self.thread = threads[0]

        step_sequence = [["// In caller_trivial_1.", "into"],
                         ["// In caller_trivial_2.", "into"],
                         ["// In inline_trivial_1.", "into"]]
        self.run_step_sequence(step_sequence)

        # Then call some trivial function, and make sure we end up back where
        # we were in the inlined call stack:

        frame = self.thread.GetFrameAtIndex(0)
        before_line_entry = frame.GetLineEntry()
        value = frame.EvaluateExpression("function_to_call()")
        after_line_entry = frame.GetLineEntry()

        self.assertEqual(
            before_line_entry.GetLine(), after_line_entry.GetLine(),
            "Line entry before and after function calls are the same.")

        # Now make sure stepping OVER in the middle of the stack works, and
        # then check finish from the inlined frame:

        step_sequence = [["// At increment in inline_trivial_1.", "over"],
                         ["// At increment in caller_trivial_2.", "out"]]
        self.run_step_sequence(step_sequence)

        # Now run to the place in main just before the first call to
        # caller_ref_1:

        break_3_in_main = target.BreakpointCreateBySourceRegex(
            '// At first call of caller_ref_1 in main.', self.main_source_spec)
        self.assertTrue(break_3_in_main, VALID_BREAKPOINT)

        threads = lldbutil.continue_to_breakpoint(
            self.process, break_3_in_main)
        self.assertEqual(
            len(threads), 1,
            "Successfully ran to call site of first caller_ref_1 call.")
        self.thread = threads[0]

        step_sequence = [["// In caller_ref_1.", "into"],
                         ["// In caller_ref_2.", "into"],
                         ["// In inline_ref_1.", "into"],
                         ["// In inline_ref_2.", "into"],
                         ["// In called_by_inline_ref.", "into"],
                         ["// In inline_ref_2.", "out"],
                         ["// In inline_ref_1.", "out"],
                         ["// At increment in inline_ref_1.", "over"],
                         ["// In caller_ref_2.", "out"],
                         ["// At increment in caller_ref_2.", "over"]]
        self.run_step_sequence(step_sequence)

    def inline_stepping_step_over(self):
        """Use Python APIs to test stepping over and hitting breakpoints."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        break_1_in_main = target.BreakpointCreateBySourceRegex(
            '// At second call of caller_ref_1 in main.', self.main_source_spec)
        self.assertTrue(break_1_in_main, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, break_1_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        self.thread = threads[0]

        step_sequence = [["// In caller_ref_1.", "into"],
                         ["// In caller_ref_2.", "into"],
                         ["// At increment in caller_ref_2.", "over"]]
        self.run_step_sequence(step_sequence)

    def step_in_template(self):
        """Use Python APIs to test stepping in to templated functions."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        break_1_in_main = target.BreakpointCreateBySourceRegex(
            '// Call max_value template', self.main_source_spec)
        self.assertTrue(break_1_in_main, VALID_BREAKPOINT)

        break_2_in_main = target.BreakpointCreateBySourceRegex(
            '// Call max_value specialized', self.main_source_spec)
        self.assertTrue(break_2_in_main, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(self.process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, break_1_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        self.thread = threads[0]

        step_sequence = [["// In max_value template", "into"]]
        self.run_step_sequence(step_sequence)

        threads = lldbutil.continue_to_breakpoint(
            self.process, break_2_in_main)
        self.assertEqual(
            len(threads),
            1,
            "Successfully ran to call site of second caller_trivial_1 call.")
        self.thread = threads[0]

        step_sequence = [["// In max_value specialized", "into"]]
        self.run_step_sequence(step_sequence)
