"""
Test that commands do not try and hold on to stale CommandInterpreters in a multiple debuggers scenario
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MultipleDebuggersCommandsTestCase(TestBase):

    @no_debug_info_test
    def test_multipledebuggers_commands(self):
        """Test that commands do not try and hold on to stale CommandInterpreters in a multiple debuggers scenario"""
        source_init_files = False
        magic_text = "The following commands may relate to 'env'"

        debugger_1 = lldb.SBDebugger.Create(source_init_files)
        interpreter_1 = debugger_1.GetCommandInterpreter()

        retobj = lldb.SBCommandReturnObject()
        interpreter_1.HandleCommand("apropos env", retobj)
        self.assertTrue(
            magic_text in str(retobj),
            "[interpreter_1]: the output does not contain the correct words")

        if self.TraceOn():
            print(str(retobj))

        lldb.SBDebugger.Destroy(debugger_1)

        # now do this again with a different debugger - we shouldn't crash

        debugger_2 = lldb.SBDebugger.Create(source_init_files)
        interpreter_2 = debugger_2.GetCommandInterpreter()

        retobj = lldb.SBCommandReturnObject()
        interpreter_2.HandleCommand("apropos env", retobj)
        self.assertTrue(
            magic_text in str(retobj),
            "[interpreter_2]: the output does not contain the correct words")

        if self.TraceOn():
            print(str(retobj))

        lldb.SBDebugger.Destroy(debugger_2)
