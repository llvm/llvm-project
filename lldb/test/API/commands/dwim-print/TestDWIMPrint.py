"""
Test dwim-print with variables, variable paths, and expressions.
"""

import re
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def _run_cmd(self, cmd: str) -> str:
        """Run the given lldb command and return its output."""
        result = lldb.SBCommandReturnObject()
        self.ci.HandleCommand(cmd, result)
        return result.GetOutput().rstrip()

    VAR_IDENT = re.compile(r"(?:\$\d+|\w+) = ")

    def _strip_result_var(self, string: str) -> str:
        """
        Strip (persistent) result variables (ex '$0 = ', or 'someVar = ', etc).

        This allows for using the output of `expression`/`frame variable`, to
        compare it to `dwim-print` output, which disables result variables.
        """
        return self.VAR_IDENT.subn("", string, 1)[0]

    def _expect_cmd(
        self,
        dwim_cmd: str,
        actual_cmd: str,
    ) -> None:
        """Run dwim-print and verify the output against the expected command."""
        # Resolve the dwim-print command to either `expression` or `frame variable`.
        substitute_cmd = dwim_cmd.replace("dwim-print", actual_cmd, 1)
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.ResolveCommand(substitute_cmd, result)
        self.assertTrue(result.Succeeded(), result.GetError())

        resolved_cmd = result.GetOutput()
        if actual_cmd == "frame variable":
            resolved_cmd = resolved_cmd.replace(" -- ", " ", 1)

        resolved_cmd_output = self._run_cmd(resolved_cmd)
        dwim_cmd_output = self._strip_result_var(resolved_cmd_output)

        # Verify dwim-print chose the expected command.
        self.runCmd("settings set dwim-print-verbosity full")

        self.expect(dwim_cmd, substrs=[
            f"note: ran `{resolved_cmd}`",
            dwim_cmd_output,
        ])

    def test_variables(self):
        """Test dwim-print with variables."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        vars = ("argc", "argv")
        for var in vars:
            self._expect_cmd(f"dwim-print {var}", "frame variable")

    def test_variable_paths(self):
        """Test dwim-print with variable path expressions."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        exprs = ("&argc", "*argv", "argv[0]")
        for expr in exprs:
            self._expect_cmd(f"dwim-print {expr}", "expression")

    def test_expressions(self):
        """Test dwim-print with expressions."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        exprs = ("argc + 1", "(void)argc", "(int)abs(argc)")
        for expr in exprs:
            self._expect_cmd(f"dwim-print {expr}", "expression")

    def test_dummy_target_expressions(self):
        """Test dwim-print's ability to evaluate expressions without a target."""
        self._expect_cmd("dwim-print 1 + 2", "expression")

    def test_gdb_format(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self._expect_cmd(f"dwim-print/x argc", "frame variable")
        self._expect_cmd(f"dwim-print/x argc + 1", "expression")

    def test_format_flags(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self._expect_cmd(f"dwim-print -fx -- argc", "frame variable")
        self._expect_cmd(f"dwim-print -fx -- argc + 1", "expression")

    def test_display_flags(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self._expect_cmd(f"dwim-print -T -- argc", "frame variable")
        self._expect_cmd(f"dwim-print -T -- argc + 1", "expression")

    def test_expression_language(self):
        """Test that the language flag doesn't affect the choice of command."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self._expect_cmd(f"dwim-print -l c++ -- argc", "frame variable")
        self._expect_cmd(f"dwim-print -l c++ -- argc + 1", "expression")

    def test_empty_expression(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        error_msg = "error: 'dwim-print' takes a variable or expression"
        self.expect(f"dwim-print", error=True, startstr=error_msg)
        self.expect(f"dwim-print -- ", error=True, startstr=error_msg)

    def test_nested_values(self):
        """Test dwim-print with nested values (structs, etc)."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))
        self.runCmd("settings set auto-one-line-summaries false")
        self._expect_cmd(f"dwim-print s", "frame variable")
        self._expect_cmd(f"dwim-print (struct Structure)s", "expression")

    def test_summary_strings(self):
        """Test dwim-print with nested values (structs, etc)."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))
        self.runCmd("settings set auto-one-line-summaries false")
        self.runCmd("type summary add -e -s 'stub summary' Structure")
        self._expect_cmd(f"dwim-print s", "frame variable")
        self._expect_cmd(f"dwim-print (struct Structure)s", "expression")
