"""
Test that lldb command "command source" works correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CommandSourceTestCase(TestBase):
    @no_debug_info_test
    def test_command_source(self):
        """Test that lldb command "command source" works correctly."""

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.runCmd("command source .lldb")
        self.check_results()

    @no_debug_info_test
    def test_command_source_relative(self):
        """Test that lldb command "command source" works correctly with relative paths."""

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.runCmd("command source commands2.txt")
        self.check_results()

    def check_results(self, failure=False):
        # Python should evaluate "my.date()" successfully.
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()
        command_interpreter.HandleCommand("script my.date()", result)

        import datetime

        if failure:
            self.expect(
                result.GetOutput(),
                "script my.date() runs successfully",
                exe=False,
                error=True,
            )
        else:
            self.expect(
                result.GetOutput(),
                "script my.date() runs successfully",
                exe=False,
                substrs=[str(datetime.date.today())],
            )

    @no_debug_info_test
    def test_command_source_relative_error(self):
        """Test that 'command source -C' gives an error for a relative path"""
        source_dir = self.getSourceDir()
        result = lldb.SBCommandReturnObject()
        self.runCmd("command source --stop-on-error 1 not-relative.txt")
        self.check_results(failure=True)
