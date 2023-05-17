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

    def test_backticks_in_parsed_cmd_argument(self):
        """ break list is a parsed command, use a variable for the breakpoint number
            and make sure that and the direct use of the ID get the same result. """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        # Make a second breakpoint so that if the backtick part -> nothing we'll print too much:
        # It doesn't need to resolve to anything.
        dummy_bkpt = target.BreakpointCreateByName("dont_really_care_if_this_exists")
        
        bkpt_id = bkpt.GetID()
        self.runCmd(f"expr int $number = {bkpt_id}")
        direct_result = lldb.SBCommandReturnObject()
        backtick_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand(f"break list {bkpt_id}", direct_result)
        self.assertTrue(direct_result.Succeeded(), "Break list with id works")
        interp.HandleCommand("break list `$number`", backtick_result)
        self.assertTrue(direct_result.Succeeded(), "Break list with backtick works")
        self.assertEqual(direct_result.GetOutput(), backtick_result.GetOutput(), "Output is the same")

    def test_backticks_in_parsed_cmd_option(self):
        # The script interpreter is a raw command, so try that one:
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        self.runCmd(f"expr int $number = 2")
        direct_result = lldb.SBCommandReturnObject()
        backtick_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand(f"memory read --count 2 argv", direct_result)
        self.assertTrue(direct_result.Succeeded(), "memory read with direct count works")
        interp.HandleCommand("memory read --count `$number` argv", backtick_result)
        self.assertTrue(direct_result.Succeeded(), "memory read with backtick works")
        self.assertEqual(direct_result.GetOutput(), backtick_result.GetOutput(), "Output is the same")

    def test_backticks_in_raw_cmd(self):
        # The script interpreter is a raw command, so try that one:
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        argc_valobj = thread.frames[0].FindVariable("argc")
        self.assertTrue(argc_valobj.GetError().Success(), "Made argc valobj")
        argc_value = argc_valobj.GetValueAsUnsigned(0)
        self.assertNotEqual(argc_value, 0, "Got a value for argc")
        result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()
        interp.HandleCommand(f"script {argc_value} - `argc`", result)
        self.assertTrue(result.Succeeded(), "Command succeeded")
        fixed_output = result.GetOutput().rstrip()
        self.assertEqual("0", fixed_output, "Substitution worked")

        
