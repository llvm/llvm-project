"""
Test Unified Section List merging.
"""

import os
import shutil

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

    def test_unified_section_list_overwrite_larger_section(self):
        """
        Test the merging of an ELF file with another ELF File where all the new sections are bigger, validating we
        overwrite .comment from SHT_NOBITS to the new SHT_PROGBITS section and the smaller .text with the larger
        .text
        """
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("main.yaml", exe)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        main_exe_module = target.GetModuleAtIndex(0)

        # First we verify out .text section is the expected BEC0FFEE
        text_before_merge = main_exe_module.FindSection(".text")
        self.assertTrue(text_before_merge.IsValid())
        error = lldb.SBError()
        section_content = text_before_merge.data.ReadRawData(
            error, 0, text_before_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content, bytes.fromhex("BEC0FFEE"))

        # .comment in main.yaml should be SHT_NOBITS, and size 0
        comment_before_merge = main_exe_module.FindSection(".comment")
        self.assertTrue(comment_before_merge.IsValid())
        self.assertEqual(comment_before_merge.data.size, 0)

        # yamlize the main.largertext.yaml and force symbol loading
        debug_info = self.getBuildArtifact("a.out.debug")
        self.yaml2obj("main.largertext.yaml", debug_info)

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand(f"target symbols add {debug_info}", res)
        self.assertTrue(res.Succeeded())

        # verify we took the larger .text section
        main_exe_module_after_merge = target.GetModuleAtIndex(0)
        text_after_merge = main_exe_module_after_merge.FindSection(".text")
        self.assertTrue(text_after_merge.IsValid())
        self.assertGreater(text_after_merge.data.size, text_before_merge.data.size)
        section_content_after_merge = text_after_merge.data.ReadRawData(
            error, 0, text_after_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content_after_merge, bytes.fromhex("BEC0FFEEEEFF0CEB"))

        # in main.largertext.yaml comment is not SHT_NOBITS, and so we should see
        # the size > 0 and equal to BAADF00D
        comment_after_merge = main_exe_module_after_merge.FindSection(".comment")
        self.assertTrue(comment_after_merge.IsValid())
        comment_content_after_merge = comment_after_merge.data.ReadRawData(
            error, 0, comment_after_merge.data.size
        )

        self.assertTrue(error.Success())
        self.assertEqual(comment_content_after_merge, bytes.fromhex("BAADF00D"))

    def test_unified_section_list_overwrite_smaller_section(self):
        """
        Test the merging of an ELF file with another ELF File where all the existing sections are bigger, validating we don't
        overwrite with the SHT_NOBITS for .comment or the smaller .text section.
        """
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("main.largertext.yaml", exe)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        main_exe_module = target.GetModuleAtIndex(0)

        # Same as above test but inverse, verify our larger .text section
        # is the expected BEC0FFEE palindrome
        text_before_merge = main_exe_module.FindSection(".text")
        self.assertTrue(text_before_merge.IsValid())
        error = lldb.SBError()
        section_content = text_before_merge.data.ReadRawData(
            error, 0, text_before_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content, bytes.fromhex("BEC0FFEEEEFF0CEB"))

        # Comment is SHT_PROGBITS on the larger yaml and should remain
        # the same after merge.
        comment_before_merge = main_exe_module.FindSection(".comment")
        self.assertTrue(comment_before_merge.IsValid())
        comment_content = comment_before_merge.data.ReadRawData(
            error, 0, comment_before_merge.data.size
        )

        self.assertTrue(error.Success())
        self.assertEqual(comment_content, bytes.fromhex("BAADF00D"))

        debug_info = self.getBuildArtifact("a.out.debug")
        self.yaml2obj("main.yaml", debug_info)

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand(f"target symbols add {debug_info}", res)
        self.assertTrue(res.Succeeded())

        # Verify we didn't replace the sections after merge.s
        main_exe_module_after_merge = target.GetModuleAtIndex(0)
        text_after_merge = main_exe_module_after_merge.FindSection(".text")
        self.assertTrue(text_after_merge.IsValid())
        self.assertEqual(text_after_merge.data.size, text_before_merge.data.size)
        section_content_after_merge = text_after_merge.data.ReadRawData(
            error, 0, text_after_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content_after_merge, bytes.fromhex("BEC0FFEEEEFF0CEB"))

        comment_after_merge = main_exe_module_after_merge.FindSection(".comment")
        self.assertTrue(comment_after_merge.IsValid())
        comment_content_after_merge = comment_after_merge.data.ReadRawData(
            error, 0, comment_after_merge.data.size
        )

        self.assertTrue(error.Success())
        self.assertEqual(comment_content_after_merge, bytes.fromhex("BAADF00D"))

    def test_unified_section_list_overwrite_mixed_merge(self):
        """
        Test the merging of an ELF file with another ELF File where the lhs has a larger .comment section
        and the RHS has a larger .text section.
        """
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("main.largercomment.yaml", exe)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        main_exe_module = target.GetModuleAtIndex(0)

        # Verify we have the expected smaller BEC0FFEE
        text_before_merge = main_exe_module.FindSection(".text")
        self.assertTrue(text_before_merge.IsValid())
        error = lldb.SBError()
        section_content = text_before_merge.data.ReadRawData(
            error, 0, text_before_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content, bytes.fromhex("BEC0FFEE"))

        # Verify we have the larger palindromic comment
        comment_before_merge = main_exe_module.FindSection(".comment")
        self.assertTrue(comment_before_merge.IsValid())
        comment_content = comment_before_merge.data.ReadRawData(
            error, 0, comment_before_merge.data.size
        )

        self.assertTrue(error.Success())
        self.assertEqual(comment_content, bytes.fromhex("BAADF00DF00DBAAD"))

        debug_info = self.getBuildArtifact("a.out.debug")
        self.yaml2obj("main.largertext.yaml", debug_info)

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand(f"target symbols add {debug_info}", res)
        self.assertTrue(res.Succeeded())

        # Verify we replaced .text
        main_exe_module_after_merge = target.GetModuleAtIndex(0)
        text_after_merge = main_exe_module_after_merge.FindSection(".text")
        self.assertTrue(text_after_merge.IsValid())
        section_content_after_merge = text_after_merge.data.ReadRawData(
            error, 0, text_after_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content_after_merge, bytes.fromhex("BEC0FFEEEEFF0CEB"))

        # Verify .comment is still the same.
        comment_after_merge = main_exe_module_after_merge.FindSection(".comment")
        self.assertTrue(comment_after_merge.IsValid())
        comment_content_after_merge = comment_after_merge.data.ReadRawData(
            error, 0, comment_after_merge.data.size
        )

        self.assertTrue(error.Success())
        self.assertEqual(comment_content_after_merge, bytes.fromhex("BAADF00DF00DBAAD"))

    def test_unified_section_list_overwrite_equal_size(self):
        """
        Test the merging of an ELF file with an ELF file with sections of the same size with different values
        .text
        """
        exe = self.getBuildArtifact("a.out")
        self.yaml2obj("main.yaml", exe)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        main_exe_module = target.GetModuleAtIndex(0)

        # First we verify out .text section is the expected BEC0FFEE
        text_before_merge = main_exe_module.FindSection(".text")
        self.assertTrue(text_before_merge.IsValid())
        error = lldb.SBError()
        section_content = text_before_merge.data.ReadRawData(
            error, 0, text_before_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content, bytes.fromhex("BEC0FFEE"))

        # .comment in main.yaml should be SHT_NOBITS, and size 0
        comment_before_merge = main_exe_module.FindSection(".comment")
        self.assertTrue(comment_before_merge.IsValid())
        self.assertEqual(comment_before_merge.data.size, 0)

        # yamlize the main with the .text reversed from BEC0FFEE
        # to EEFF0CEB. We should still keep our .text with BEC0FFEE
        debug_info = self.getBuildArtifact("a.out.debug")
        self.yaml2obj("main.reversedtext.yaml", debug_info)

        ci = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand(f"target symbols add {debug_info}", res)
        self.assertTrue(res.Succeeded())

        # verify .text did not change
        main_exe_module_after_merge = target.GetModuleAtIndex(0)
        text_after_merge = main_exe_module_after_merge.FindSection(".text")
        self.assertTrue(text_after_merge.IsValid())
        section_content_after_merge = text_after_merge.data.ReadRawData(
            error, 0, text_after_merge.data.size
        )
        self.assertTrue(error.Success())
        self.assertEqual(section_content_after_merge, bytes.fromhex("BEC0FFEE"))

        # verify comment did not change
        comment_afer_merge = main_exe_module_after_merge.FindSection(".comment")
        self.assertTrue(comment_afer_merge.IsValid())
        self.assertEqual(comment_afer_merge.data.size, 0)
