"""
Test lldb breakpoint command for CPP methods & functions in a namespace.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class CPPBreakpointTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "breakpoints")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.buildDsym()
        self.cpp_breakpoints()

    @dwarf_test
    def test_with_dwarf(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.buildDwarf()
        self.cpp_breakpoints()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def cpp_breakpoints (self):
        """Test a sequence of breakpoint command add, list, and delete."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        a_out_module = lldb.SBFileSpecList()
        a_out_module.Append(lldb.SBFileSpec(exe))

        nested_comp_unit = lldb.SBFileSpecList()
        nested_comp_unit.Append (lldb.SBFileSpec("nested.cpp"))

        # First provide ONLY the method name.  This should get everybody...
        auto_break = target.BreakpointCreateByName ("Function",
                                                    lldb.eFunctionNameTypeAuto,
                                                    a_out_module,
                                                    nested_comp_unit)
        self.assertTrue (auto_break.GetNumLocations() == 5)

        # Now add the Baz class specifier.  This should get the version contained in Bar,
        # AND the one contained in ::
        auto_break = target.BreakpointCreateByName ("Baz::Function",
                                                    lldb.eFunctionNameTypeAuto,
                                                    a_out_module,
                                                    nested_comp_unit)
        self.assertTrue (auto_break.GetNumLocations() == 2)

        # Then add the Bar::Baz specifier.  This should get the version contained in Bar only
        auto_break = target.BreakpointCreateByName ("Bar::Baz::Function",
                                                    lldb.eFunctionNameTypeAuto,
                                                    a_out_module,
                                                    nested_comp_unit)
        self.assertTrue (auto_break.GetNumLocations() == 1)

        plain_method_break = target.BreakpointCreateByName ("Function", 
                                                            lldb.eFunctionNameTypeMethod,
                                                            a_out_module,
                                                            nested_comp_unit)
        self.assertTrue (plain_method_break.GetNumLocations() == 3)

        plain_method_break = target.BreakpointCreateByName ("Baz::Function", 
                                                            lldb.eFunctionNameTypeMethod,
                                                            a_out_module,
                                                            nested_comp_unit)
        self.assertTrue (plain_method_break.GetNumLocations() == 2)

        plain_method_break = target.BreakpointCreateByName ("Bar::Baz::Function", 
                                                            lldb.eFunctionNameTypeMethod,
                                                            a_out_module,
                                                            nested_comp_unit)
        self.assertTrue (plain_method_break.GetNumLocations() == 1)

        plain_method_break = target.BreakpointCreateByName ("Function", 
                                                            lldb.eFunctionNameTypeBase,
                                                            a_out_module,
                                                            nested_comp_unit)
        self.assertTrue (plain_method_break.GetNumLocations() == 2)

        plain_method_break = target.BreakpointCreateByName ("Bar::Function", 
                                                            lldb.eFunctionNameTypeBase,
                                                            a_out_module,
                                                            nested_comp_unit)
        self.assertTrue (plain_method_break.GetNumLocations() == 1)

        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
