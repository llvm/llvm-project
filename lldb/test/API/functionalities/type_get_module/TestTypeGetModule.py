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

        self.assertTrue(result.IsValid())
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

        self.assertTrue(result.IsValid())
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

        self.assertTrue(result.IsValid())
        return result

    def test(self):
        self.build()
        target  = lldbutil.run_to_breakpoint_make_target(self)
        exe_module = self.find_module(target, 'a.out')

        num_comp_units = exe_module.GetNumCompileUnits()
        self.assertGreaterEqual(num_comp_units, 3)

        comp_unit = self.find_comp_unit(exe_module, 'compile_unit1.c')
        cu_type = self.find_type(comp_unit.GetTypes(), 'compile_unit1_type')
        self.assertTrue(exe_module == cu_type.GetModule())
        
        comp_unit = self.find_comp_unit(exe_module, 'compile_unit2.c')
        cu_type = self.find_type(comp_unit.GetTypes(), 'compile_unit2_type')
        self.assertTrue(exe_module == cu_type.GetModule())
