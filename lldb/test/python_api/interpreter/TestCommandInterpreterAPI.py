"""Test the SBCommandInterpreter APIs."""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class CommandInterpreterAPICase(TestBase):

    mydir = os.path.join("python_api", "interpreter")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_process_launch_api(self):
        """Test the SBCommandInterpreter APIs."""
        self.buildDsym()
        self.command_interpreter_api()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_process_launch_api(self):
        """Test the SBCommandInterpreter APIs."""
        self.buildDwarf()
        self.command_interpreter_api()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number('main.c', 'Hello world.')

    def command_interpreter_api(self):
        """Test the SBCommandInterpreter APIs."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Exercise some APIs....

        self.assertTrue(ci.HasCommands())
        self.assertTrue(ci.HasAliases())
        self.assertTrue(ci.HasAliasOptions())
        self.assertTrue(ci.CommandExists("breakpoint"))
        self.assertTrue(ci.CommandExists("target"))
        self.assertTrue(ci.CommandExists("platform"))
        self.assertTrue(ci.AliasExists("file"))
        self.assertTrue(ci.AliasExists("run"))
        self.assertTrue(ci.AliasExists("bt"))

        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("breakpoint set -f main.c -l %d" % self.line, res)
        self.assertTrue(res.Succeeded())
        ci.HandleCommand("process launch", res)
        self.assertTrue(res.Succeeded())

        # Boundary conditions should not crash lldb!
        self.assertFalse(ci.CommandExists(None))
        self.assertFalse(ci.AliasExists(None))
        ci.HandleCommand(None, res)
        self.assertFalse(res.Succeeded())
        res.AppendMessage("Just appended a message.")
        res.AppendMessage(None)
        if self.TraceOn():
            print res

        process = ci.GetProcess()
        self.assertTrue(process)

        import lldbutil
        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        if self.TraceOn():
            lldbutil.print_stacktraces(process)        
                        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
