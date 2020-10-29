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

    def find_module(self, target, name):
        num_modules = target.GetNumModules()
        index = 0
        result = lldb.SBModule()
        
        while index < num_modules:
            module = target.GetModuleAtIndex(index)
            if module.GetFileSpec().GetFilename() == name:
                result = module
                break
                
            index += 1
            
        return result

    def find_comp_unit(self, exe_module, name):
        num_comp_units = exe_module.GetNumCompileUnits()
        index = 0
        result = lldb.SBCompileUnit()
        
        while index < num_comp_units:
            comp_unit = exe_module.GetCompileUnitAtIndex(index)
            if comp_unit.GetFileSpec().GetFilename() == name:
                result = comp_unit
                break
                
            index += 1
            
        return result
        
    def find_type(self, type_list, name):
        num_types = type_list.GetSize()
        index = 0
        result = lldb.SBType()
        
        while index < num_types:
            type = type_list.GetTypeAtIndex(index)
            if type.GetName() == name:
                result = type
                break
                
            index += 1
        
        return result
            
    def test(self):
        self.build()
        target  = lldbutil.run_to_breakpoint_make_target(self)
        exe_module = self.find_module(target, 'a.out')
        self.assertTrue(exe_module.IsValid())
        
        type1_name = 'compile_unit1_type'
        type2_name = 'compile_unit2_type'
        
        num_comp_units = exe_module.GetNumCompileUnits()
        self.assertEqual(num_comp_units, 3)
        
        comp_unit = self.find_comp_unit(exe_module, 'compile_unit1.c')
        self.assertTrue(comp_unit.IsValid())
        
        cu_type = self.find_type(comp_unit.GetTypes(), type1_name)
        self.assertTrue(cu_type.IsValid())
        self.assertEqual(cu_type.GetName(), type1_name)
        
        comp_unit = self.find_comp_unit(exe_module, 'compile_unit2.c')
        self.assertTrue(comp_unit.IsValid())
        
        cu_type = self.find_type(comp_unit.GetTypes(), type2_name)
        self.assertTrue(cu_type.IsValid())
        self.assertEqual(cu_type.GetName(), type2_name)
        
        type1 = target.FindFirstType(type1_name)
        self.assertTrue(type1.IsValid())
        
        type2 = target.FindFirstType(type2_name)
        self.assertTrue(type2.IsValid())
        
        self.assertTrue(exe_module == type1.GetModule() and
                        exe_module == type2.GetModule())
