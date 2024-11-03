"""
Test SBSection APIs.
"""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SectionAPITestCase(TestBase):

    @no_debug_info_test
    @skipIfXmlSupportMissing
    def test_get_target_byte_size(self):
        d = {'EXE': 'b.out'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = self.getBuildArtifact('b.out')
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # find the .data section of the main module
        mod = target.GetModuleAtIndex(0)
        data_section = None
        for s in mod.sections:
            sect_type = s.GetSectionType()
            if sect_type == lldb.eSectionTypeData:
                data_section = s
                break
            elif sect_type == lldb.eSectionTypeContainer:
                for i in range(s.GetNumSubSections()):
                    ss = s.GetSubSectionAtIndex(i)
                    sect_type = ss.GetSectionType()
                    if sect_type == lldb.eSectionTypeData:
                        data_section = ss
                        break

        self.assertIsNotNone(data_section)
        self.assertEqual(data_section.target_byte_size, 1)

    @no_debug_info_test
    @skipIfXmlSupportMissing
    def test_get_alignment(self):
        exe = self.getBuildArtifact("aligned.out")
        self.yaml2obj("aligned.yaml", exe)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # exe contains a single section aligned to 0x1000
        section = target.modules[0].sections[0]
        self.assertEqual(section.GetAlignment(), 0x1000)
        self.assertEqual(section.alignment, 0x1000)

    @no_debug_info_test
    @skipIfXmlSupportMissing
    def test_compressed_section_data(self):
        exe = self.getBuildArtifact("compressed-sections.out")
        self.yaml2obj("compressed-sections.yaml", exe)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # exe contains a single section with SHF_COMPRESSED. Check that
        # GetSectionData returns the uncompressed data and not the raw contents
        # of the section.
        section = target.modules[0].sections[0]
        section_data = section.GetSectionData().uint8s
        self.assertEqual(section_data,
                         [0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90])

