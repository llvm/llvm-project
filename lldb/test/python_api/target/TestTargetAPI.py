"""
Test SBTarget APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class TargetAPITestCase(TestBase):

    mydir = os.path.join("python_api", "target")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_find_global_variables_with_dsym(self):
        """Exercise SBTaget.FindGlobalVariables() API."""
        d = {'EXE': 'a.out'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_global_variables('a.out')

    #rdar://problem/9700873
    # Find global variable value fails for dwarf if inferior not started
    # (Was CrashTracer: [USER] 1 crash in Python at _lldb.so: lldb_private::MemoryCache::Read + 94)
    #
    # It does not segfaults now.  But for dwarf, the variable value is None if
    # the inferior process does not exist yet.  The radar has been updated.
    #@unittest232.skip("segmentation fault -- skipping")
    @python_api_test
    @dwarf_test
    def test_find_global_variables_with_dwarf(self):
        """Exercise SBTarget.FindGlobalVariables() API."""
        d = {'EXE': 'b.out'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_global_variables('b.out')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_find_functions_with_dsym(self):
        """Exercise SBTaget.FindFunctions() API."""
        d = {'EXE': 'a.out'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_functions('a.out')

    @python_api_test
    @dwarf_test
    def test_find_functions_with_dwarf(self):
        """Exercise SBTarget.FindFunctions() API."""
        d = {'EXE': 'b.out'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_functions('b.out')

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_get_description_with_dsym(self):
        """Exercise SBTaget.GetDescription() API."""
        self.buildDsym()
        self.get_description()

    @python_api_test
    @dwarf_test
    def test_get_description_with_dwarf(self):
        """Exercise SBTarget.GetDescription() API."""
        self.buildDwarf()
        self.get_description()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_launch_new_process_and_redirect_stdout_with_dsym(self):
        """Exercise SBTaget.Launch() API."""
        self.buildDsym()
        self.launch_new_process_and_redirect_stdout()

    @python_api_test
    @dwarf_test
    def test_launch_new_process_and_redirect_stdout_with_dwarf(self):
        """Exercise SBTarget.Launch() API."""
        self.buildDwarf()
        self.launch_new_process_and_redirect_stdout()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_resolve_symbol_context_with_address_with_dsym(self):
        """Exercise SBTaget.ResolveSymbolContextForAddress() API."""
        self.buildDsym()
        self.resolve_symbol_context_with_address()

    @python_api_test
    @dwarf_test
    def test_resolve_symbol_context_with_address_with_dwarf(self):
        """Exercise SBTarget.ResolveSymbolContextForAddress() API."""
        self.buildDwarf()
        self.resolve_symbol_context_with_address()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number for breakpoint 1 here.')
        self.line2 = line_number('main.c', '// Find the line number for breakpoint 2 here.')

    def find_global_variables(self, exe_name):
        """Exercise SBTaget.FindGlobalVariables() API."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        #rdar://problem/9700873
        # Find global variable value fails for dwarf if inferior not started
        # (Was CrashTracer: [USER] 1 crash in Python at _lldb.so: lldb_private::MemoryCache::Read + 94)
        #
        # Remove the lines to create a breakpoint and to start the inferior
        # which are workarounds for the dwarf case.

        breakpoint = target.BreakpointCreateByLocation('main.c', self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)
        # Make sure we hit our breakpoint:
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)

        value_list = target.FindGlobalVariables('my_global_var_of_char_type', 3)
        self.assertTrue(value_list.GetSize() == 1)
        my_global_var = value_list.GetValueAtIndex(0)
        self.DebugSBValue(my_global_var)
        self.assertTrue(my_global_var)
        self.expect(my_global_var.GetName(), exe=False,
            startstr = "my_global_var_of_char_type")
        self.expect(my_global_var.GetTypeName(), exe=False,
            startstr = "char")
        self.expect(my_global_var.GetValue(), exe=False,
            startstr = "'X'")

        # While we are at it, let's also exercise the similar SBModule.FindGlobalVariables() API.
        for m in target.module_iter():
            if m.GetFileSpec().GetDirectory() == os.getcwd() and m.GetFileSpec().GetFilename() == exe_name:
                value_list = m.FindGlobalVariables(target, 'my_global_var_of_char_type', 3)
                self.assertTrue(value_list.GetSize() == 1)
                self.assertTrue(value_list.GetValueAtIndex(0).GetValue() == "'X'")
                break

    def find_functions(self, exe_name):
        """Exercise SBTaget.FindFunctions() API."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        list = target.FindFunctions('c', lldb.eFunctionNameTypeAuto)
        self.assertTrue(list.GetSize() == 1)

        for sc in list:
            self.assertTrue(sc.GetModule().GetFileSpec().GetFilename() == exe_name)
            self.assertTrue(sc.GetSymbol().GetName() == 'c')                

    def get_description(self):
        """Exercise SBTaget.GetDescription() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        from lldbutil import get_description

        # get_description() allows no option to mean lldb.eDescriptionLevelBrief.
        desc = get_description(target)
        #desc = get_description(target, option=lldb.eDescriptionLevelBrief)
        if not desc:
            self.fail("SBTarget.GetDescription() failed")
        self.expect(desc, exe=False,
            substrs = ['a.out'])
        self.expect(desc, exe=False, matching=False,
            substrs = ['Target', 'Module', 'Breakpoint'])

        desc = get_description(target, option=lldb.eDescriptionLevelFull)
        if not desc:
            self.fail("SBTarget.GetDescription() failed")
        self.expect(desc, exe=False,
            substrs = ['a.out', 'Target', 'Module', 'Breakpoint'])


    def launch_new_process_and_redirect_stdout(self):
        """Exercise SBTaget.Launch() API with redirected stdout."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Add an extra twist of stopping the inferior in a breakpoint, and then continue till it's done.
        # We should still see the entire stdout redirected once the process is finished.
        line = line_number('main.c', '// a(3) -> c(3)')
        breakpoint = target.BreakpointCreateByLocation('main.c', line)

        # Now launch the process, do not stop at entry point, and redirect stdout to "stdout.txt" file.
        # The inferior should run to completion after "process.Continue()" call.
        error = lldb.SBError()
        process = target.Launch (self.dbg.GetListener(), None, None, None, "stdout.txt", None, None, 0, False, error)
        process.Continue()
        #self.runCmd("process status")

        # The 'stdout.txt' file should now exist.
        self.assertTrue(os.path.isfile("stdout.txt"),
                        "'stdout.txt' exists due to redirected stdout via SBTarget.Launch() API.")

        # Read the output file produced by running the program.
        with open('stdout.txt', 'r') as f:
            output = f.read()

        # Let's delete the 'stdout.txt' file as a cleanup step.
        try:
            os.remove("stdout.txt")
            pass
        except OSError:
            pass

        self.expect(output, exe=False,
            substrs = ["a(1)", "b(2)", "a(3)"])


    def resolve_symbol_context_with_address(self):
        """Exercise SBTaget.ResolveSymbolContextForAddress() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create the two breakpoints inside function 'a'.
        breakpoint1 = target.BreakpointCreateByLocation('main.c', self.line1)
        breakpoint2 = target.BreakpointCreateByLocation('main.c', self.line2)
        #print "breakpoint1:", breakpoint1
        #print "breakpoint2:", breakpoint2
        self.assertTrue(breakpoint1 and
                        breakpoint1.GetNumLocations() == 1,
                        VALID_BREAKPOINT)
        self.assertTrue(breakpoint2 and
                        breakpoint2.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line1)

        address1 = lineEntry.GetStartAddress()

        # Continue the inferior, the breakpoint 2 should be hit.
        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        #self.runCmd("process status")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        address2 = lineEntry.GetStartAddress()

        #print "address1:", address1
        #print "address2:", address2

        # Now call SBTarget.ResolveSymbolContextForAddress() with the addresses from our line entry.
        context1 = target.ResolveSymbolContextForAddress(address1, lldb.eSymbolContextEverything)
        context2 = target.ResolveSymbolContextForAddress(address2, lldb.eSymbolContextEverything)

        self.assertTrue(context1 and context2)
        #print "context1:", context1
        #print "context2:", context2

        # Verify that the context point to the same function 'a'.
        symbol1 = context1.GetSymbol()
        symbol2 = context2.GetSymbol()
        self.assertTrue(symbol1 and symbol2)
        #print "symbol1:", symbol1
        #print "symbol2:", symbol2

        from lldbutil import get_description
        desc1 = get_description(symbol1)
        desc2 = get_description(symbol2)
        self.assertTrue(desc1 and desc2 and desc1 == desc2,
                        "The two addresses should resolve to the same symbol")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
