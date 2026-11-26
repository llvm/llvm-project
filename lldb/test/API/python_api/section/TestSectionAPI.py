"""
Test SBSection APIs.
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SectionAPITestCase(TestBase):
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
    @skipIfZLIBSupportMissing
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
        self.assertEqual(section_data, [0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90])
