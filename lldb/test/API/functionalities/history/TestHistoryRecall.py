"""
Test some features of "session history" and history recall.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestHistoryRecall(TestBase):

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_history_recall(self):
        """Test the !N and !-N functionality of the command interpreter."""
        self.do_bang_N_test()

    def test_regex_history(self):
        """Test the regex commands don't add two elements to the history"""
        self.do_regex_history_test()

    def do_regex_history_test(self):
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        command = "_regexp-break foo.c:12"
        self.runCmd(command, msg="Run the regex break command", inHistory = True)
        interp.HandleCommand("session history", result, True)
        self.assertTrue(result.Succeeded(), "session history ran successfully")
        results = result.GetOutput()
        self.assertIn(command, results, "Recorded the actual command")
        self.assertNotIn("breakpoint set --file 'foo.c' --line 12", results,
                         "Didn't record the resolved command")
        
    def do_bang_N_test(self):
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("session history", result, True)
        interp.HandleCommand("platform list", result, True)

        interp.HandleCommand("!0", result, False)
        self.assertTrue(result.Succeeded(), "!0 command did not work: %s"%(result.GetError()))
        self.assertIn("session history", result.GetOutput(), "!0 didn't rerun session history")

        interp.HandleCommand("!-1", result, False)
        self.assertTrue(result.Succeeded(), "!-1 command did not work: %s"%(result.GetError()))
        self.assertIn("host:", result.GetOutput(), "!-1 didn't rerun platform list.")
