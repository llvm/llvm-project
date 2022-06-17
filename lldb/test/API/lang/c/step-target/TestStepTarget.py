"""Test the 'step target' feature."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStepTarget(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"
        self.end_line = line_number(self.main_source, "All done")

    @add_test_categories(['pyapi'])
    def get_to_start(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        break_in_main = target.BreakpointCreateBySourceRegex(
            'Break here to try targetted stepping', self.main_source_spec)
        self.assertTrue(break_in_main, VALID_BREAKPOINT)
        self.assertGreater(break_in_main.GetNumLocations(), 0, "Has locations.")

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, break_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        thread = threads[0]
        return thread

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_with_end_line(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

        thread = self.get_to_start()

        error = lldb.SBError()
        thread.StepInto("lotsOfArgs", self.end_line, error)
        frame = thread.frames[0]

        self.assertEqual(frame.name, "lotsOfArgs", "Stepped to lotsOfArgs.")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_with_end_line_bad_name(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

        thread = self.get_to_start()

        error = lldb.SBError()
        thread.StepInto("lotsOfArgssss", self.end_line, error)
        frame = thread.frames[0]
        self.assertEqual(frame.line_entry.line, self.end_line,
            "Stepped to the block end.")

    def test_with_end_line_deeper(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

        thread = self.get_to_start()

        error = lldb.SBError()
        thread.StepInto("modifyInt", self.end_line, error)
        frame = thread.frames[0]
        self.assertEqual(frame.name, "modifyInt", "Stepped to modifyInt.")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_with_command_and_block(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

        thread = self.get_to_start()

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            'thread step-in -t "lotsOfArgs" -e block', result)
        self.assertTrue(
            result.Succeeded(),
            "thread step-in command succeeded.")

        frame = thread.frames[0]
        self.assertEqual(frame.name, "lotsOfArgs", "Stepped to lotsOfArgs.")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_with_command_and_block_and_bad_name(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

        thread = self.get_to_start()

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            'thread step-in -t "lotsOfArgsssss" -e block', result)
        self.assertTrue(
            result.Succeeded(),
            "thread step-in command succeeded.")

        frame = thread.frames[0]

        self.assertEqual(frame.name, "main", "Stepped back out to main.")
        # end_line is set to the line after the containing block.  Check that
        # we got there:
        self.assertEqual(frame.line_entry.line, self.end_line,
            "Got out of the block")
