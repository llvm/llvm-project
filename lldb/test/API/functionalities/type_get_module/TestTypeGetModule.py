"""
Test that SBType returns SBModule of executable file but not
of compile unit's object file.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestTypeGetModule(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        target  = lldbutil.run_to_breakpoint_make_target(self)
        exe_module = target.GetModuleAtIndex(0)
        
        type1_name = 'compile_unit1_type'
        type2_name = 'compile_unit2_type'
        
        num_comp_units = exe_module.GetNumCompileUnits()
        self.assertEqual(num_comp_units, 3)
        
        comp_unit = exe_module.GetCompileUnitAtIndex(1)
        type_name = comp_unit.GetTypes().GetTypeAtIndex(0).GetName()
        self.assertEqual(type_name, type1_name)
        
        comp_unit = exe_module.GetCompileUnitAtIndex(2)
        type_name = comp_unit.GetTypes().GetTypeAtIndex(0).GetName()
        self.assertEqual(type_name, type2_name)
        
        type1 = target.FindFirstType(type1_name)
        self.assertTrue(type1.IsValid())
        
        type2 = target.FindFirstType(type2_name)
        self.assertTrue(type2.IsValid())
        
        self.assertTrue(exe_module == type1.GetModule() and
                        exe_module == type2.GetModule())
