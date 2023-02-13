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

    PERSISTENT_VAR = re.compile(r"\$\d+")

    def _mask_persistent_var(self, string: str) -> str:
        """
        Replace persistent result variables (ex '$0', '$1', etc) with a regex
        that matches any persistent result (r'\$\d+'). The returned string can
        be matched against other `expression` results.
        """
        before, after = self.PERSISTENT_VAR.split(string, maxsplit=1)
        return re.escape(before) + r"\$\d+" + re.escape(after)

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

        expected_output = self._run_cmd(resolved_cmd)

        # Verify dwim-print chose the expected command.
        self.runCmd("settings set dwim-print-verbosity full")
        substrs = [f"note: ran `{resolved_cmd}`"]
        patterns = []

        if actual_cmd == "expression" and self.PERSISTENT_VAR.search(expected_output):
            patterns.append(self._mask_persistent_var(expected_output))
        else:
            substrs.append(expected_output)

        self.expect(dwim_cmd, substrs=substrs, patterns=patterns)

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
