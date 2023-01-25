"""
Test that an alias can contain active backticks
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestBackticksInAlias(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_backticks_in_alias(self):
        """Test that an alias can contain active backticks."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        interp = self.dbg.GetCommandInterpreter();
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand('command alias _test-argv-cmd expression -Z \`argc\` -- argv', result)
        self.assertCommandReturn(result, "Made the alias")
        interp.HandleCommand("_test-argv-cmd", result)
        self.assertCommandReturn(result, "The alias worked")

        # Now try a harder case where we create this using an alias:
        interp.HandleCommand('command alias _test-argv-parray-cmd parray \`argc\` argv', result)
        self.assertCommandReturn(result, "Made the alias")
        interp.HandleCommand("_test-argv-parray-cmd", result)
        self.assertFalse(result.Succeeded(), "CommandAlias::Desugar currently fails if a alias substitutes %N arguments in another alias")

        
