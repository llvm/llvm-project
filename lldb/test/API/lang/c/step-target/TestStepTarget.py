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

    @add_test_categories(["pyapi"])
    def get_to_start(self):
        self.build()

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Break here to try targetted stepping", self.main_source_spec
        )

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
        self.assertEqual(
            frame.line_entry.line, self.end_line, "Stepped to the block end."
        )

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
        self.do_command_and_block()
        self.do_command_and_block(True)

    def do_command_and_block(self, use_regexp_step=False):
        thread = self.get_to_start()

        if use_regexp_step:
            self.expect("s lotsOfArgs")
        else:
            self.expect('thread step-in -t "lotsOfArgs" -e block')

        frame = thread.frames[0]
        self.assertEqual(frame.name, "lotsOfArgs", "Stepped to lotsOfArgs.")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr32343")
    def test_with_command_and_block_and_bad_name(self):
        """Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""
        self.do_with_command_and_block_and_bad_name()
        self.do_with_command_and_block_and_bad_name(True)

    def do_with_command_and_block_and_bad_name(self, use_regexp_step=False):
        thread = self.get_to_start()

        if use_regexp_step:
            self.expect("s lotsOfArgsssss")
        else:
            self.expect('thread step-in -t "lotsOfArgsssss" -e block')

        frame = thread.frames[0]

        self.assertEqual(frame.name, "main", "Stepped back out to main.")
        # end_line is set to the line after the containing block.  Check that
        # we got there:
        self.assertEqual(frame.line_entry.line, self.end_line, "Got out of the block")
