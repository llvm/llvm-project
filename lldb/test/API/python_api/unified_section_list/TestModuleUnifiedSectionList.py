"""
Test Unified Section List merging.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import symbol_type_to_str


class ModuleUnifiedSectionList(TestBase):

    @skipUnlessPlatform(["linux", "freebsd", "netbsd"])
    def test_unified_section_list(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        debug_info = self.getBuildArtifact("a.out.debug")
        new_dir = os.path.join(os.path.dirname(debug_info), "new_dir")
        os.mkdir(new_dir)
        renamed_debug_info = os.path.join(new_dir, "renamed.debug")
        os.rename(debug_info, renamed_debug_info)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertGreater(target.GetNumModules(), 0)

        main_exe_module = target.GetModuleAtIndex(0)
        eh_frame = main_exe_module.FindSection(".eh_frame")
        self.assertTrue(eh_frame.IsValid())
        self.assertGreater(eh_frame.size, 0)

        # Should be stripped in main executable.
        debug_info_section = main_exe_module.FindSection(".debug_info")
        self.assertFalse(debug_info_section.IsValid())

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand(f"target symbols add {renamed_debug_info}", res)
        self.assertTrue(res.Succeeded())

        # Should be stripped in .debuginfo but be present in main executable.
        main_exe_module = target.GetModuleAtIndex(0)
        eh_frame = main_exe_module.FindSection(".eh_frame")
        self.assertTrue(eh_frame.IsValid())
        self.assertGreater(eh_frame.size, 0)

        # Should be unified and both sections should have contents.
        debug_info_section = main_exe_module.FindSection(".debug_info")
        self.assertTrue(debug_info_section.IsValid())
        self.assertGreater(debug_info_section.file_size, 0)
