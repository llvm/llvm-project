"""
Test SBSymbolContext APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SymbolContextAPITestCase(TestBase):

    mydir = os.path.join("python_api", "symbol-context")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym(self):
        """Exercise SBSymbolContext API extensively."""
        self.buildDsym()
        self.symbol_context()

    @python_api_test
    @dwarf_test
    def test_with_dwarf(self):
        """Exercise SBSymbolContext API extensively."""
        self.buildDwarf()
        self.symbol_context()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Find the line number of function "c" here.')

    def symbol_context(self):
        """Get an SBSymbolContext object and call its many methods."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line.
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
        self.assertTrue(context)

        # Get the description of this module.
        module = context.GetModule()
        desc = lldbutil.get_description(module)
        self.expect(desc, "The module should match", exe=False,
            substrs = [os.path.join(self.mydir, 'a.out')])

        compileUnit = context.GetCompileUnit()
        self.expect(str(compileUnit), "The compile unit should match", exe=False,
            substrs = [os.path.join(self.mydir, 'main.c')])

        function = context.GetFunction()
        self.assertTrue(function)
        #print "function:", function

        block = context.GetBlock()
        self.assertTrue(block)
        #print "block:", block

        lineEntry = context.GetLineEntry()
        #print "line entry:", lineEntry
        self.expect(lineEntry.GetFileSpec().GetDirectory(), "The line entry should have the correct directory",
                    exe=False,
            substrs = [self.mydir])
        self.expect(lineEntry.GetFileSpec().GetFilename(), "The line entry should have the correct filename",
                    exe=False,
            substrs = ['main.c'])
        self.assertTrue(lineEntry.GetLine() == self.line,
                        "The line entry's line number should match ")

        symbol = context.GetSymbol()
        self.assertTrue(function.GetName() == symbol.GetName() and symbol.GetName() == 'c',
                        "The symbol name should be 'c'")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
