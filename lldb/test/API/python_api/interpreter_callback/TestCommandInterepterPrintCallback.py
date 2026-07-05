import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CommandInterepterPrintCallbackTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def run_command_interpreter_with_output_file(self, out_filename, input_str):
        with open(out_filename, "w") as f:
            self.dbg.SetOutputFileHandle(f, False)
            self.dbg.SetInputString(input_str)
            opts = lldb.SBCommandInterpreterRunOptions()
            self.dbg.RunCommandInterpreter(True, False, opts, 0, False, False)

    def test_command_interpreter_print_callback(self):
        """Test the command interpreter print callback."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.c")
        )

        out_filename = self.getBuildArtifact("output")
        ci = self.dbg.GetCommandInterpreter()
        called = False

        # The string we'll be looking for in the command output.
        needle = "Show a list of all debugger commands"

        # Test registering a callback that handles the printing. Make sure the
        # result is passed to the callback and that we don't print the result.
        def handling_callback(return_object):
            nonlocal called
            called = True
            self.assertEqual("help help", return_object.GetCommand())
            self.assertIn(needle, return_object.GetOutput())
            return lldb.eCommandReturnObjectPrintCallbackHandled

        ci.SetPrintCallback(handling_callback)
        self.assertFalse(called)
        self.run_command_interpreter_with_output_file(out_filename, "help help\n")
        with open(out_filename, "r") as f:
            self.assertNotIn(needle, f.read())

        # Test registering a callback that defers the printing to lldb. Make
        # sure the result is passed to the callback and that the result is
        # printed by lldb.
        def non_handling_callback(return_object):
            nonlocal called
            called = True
            self.assertEqual("he help", return_object.GetCommand())
            self.assertIn(needle, return_object.GetOutput())
            return lldb.eCommandReturnObjectPrintCallbackSkipped

        called = False
        ci.SetPrintCallback(non_handling_callback)
        self.assertFalse(called)
        self.run_command_interpreter_with_output_file(out_filename, "he help\n")
        self.assertTrue(called)

        with open(out_filename, "r") as f:
            self.assertIn(needle, f.read())
