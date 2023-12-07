"""
Test absolute symbols in ELF files to make sure they don't create sections and
to verify that symbol values and size can still be accessed via SBSymbol APIs.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class TestAbsoluteSymbol(TestBase):
    @no_debug_info_test
    def test_absolute_symbol(self):
        """
        Load an ELF file that contains two symbols:
        - "absolute" which is a symbol with the section SHN_ABS
        - "main" which is a code symbol in .text

        Index   st_name    st_value           st_size            st_info                             st_other st_shndx Name
        ======= ---------- ------------------ ------------------ ----------------------------------- -------- -------- ===========================
        [    0] 0x00000000 0x0000000000000000 0x0000000000000000 0x00 (STB_LOCAL      STT_NOTYPE   ) 0x00            0
        [    1] 0x00000001 0x0000000000001000 0x0000000000000004 0x12 (STB_GLOBAL     STT_FUNC     ) 0x00            1 main
        [    2] 0x00000006 0xffffffff80000000 0x0000000000000009 0x10 (STB_GLOBAL     STT_NOTYPE   ) 0x00      SHN_ABS absolute

        We used to create sections for symbols whose section ID was SHN_ABS
        and this caused problems as the new sections could interfere with
        with address resolution. Absolute symbols' values are not addresses
        and should not be treated this way.

        New APIs were added to SBSymbol to allow access to the raw integer
        value and size of symbols so symbols whose value was not an address
        could be accessed. Prior to this commit, you could only call:

        SBAddress SBSymbol::GetStartAddress()
        SBAddress SBSymbol::GetEndAddress()

        If the symbol's value was not an address, you couldn't access the
        raw value because the above accessors would return invalid SBAddress
        objects if the value wasn't an address. New APIs were added for this:

        uint64_t SBSymbol::GetValue()
        uint64_t SBSymbol::GetSize();
        """
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "absolute.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("a.out")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        module = target.modules[0]

        # Make sure the 'main' symbol is valid and has address values. Also make
        # sure we can access the raw file address and size via the new APIS.
        symbol = module.FindSymbol("main")
        self.assertTrue(symbol.IsValid())
        self.assertTrue(symbol.GetStartAddress().IsValid())
        self.assertTrue(symbol.GetEndAddress().IsValid())
        self.assertEqual(symbol.GetValue(), 0x1000)
        self.assertEqual(symbol.GetSize(), 0x4)

        # Make sure the 'absolute' symbol is valid and has no address values.
        # Also make sure we can access the raw file address and size via the new
        # APIS.
        symbol = module.FindSymbol("absolute")
        self.assertTrue(symbol.IsValid())
        self.assertFalse(symbol.GetStartAddress().IsValid())
        self.assertFalse(symbol.GetEndAddress().IsValid())
        self.assertEqual(symbol.GetValue(), 0xFFFFFFFF80000000)
        self.assertEqual(symbol.GetSize(), 9)

        # Make sure no sections were created for the absolute symbol with a
        # prefix of ".absolute." followed by the symbol name as they interfere
        # with address lookups if they are treated like real sections.
        for section in module.sections:
            self.assertNotEqual(section.GetName(), ".absolute.absolute")
