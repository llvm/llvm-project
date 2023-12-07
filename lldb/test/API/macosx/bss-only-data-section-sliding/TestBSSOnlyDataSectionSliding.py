"""Test that we a BSS-data only DATA segment is slid with other segments."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBSSOnlyDataSectionSliding(TestBase):
    @skipUnlessDarwin
    def test_with_python_api(self):
        """Test that we get thread names when interrupting a process."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe, "", "", False, lldb.SBError())
        self.assertTrue(target, VALID_TARGET)

        module = target.modules[0]
        self.assertTrue(module.IsValid())
        data_sect = module.section["__DATA"]
        self.assertTrue(data_sect.IsValid())

        target.SetModuleLoadAddress(module, 0x170000000)
        self.assertEqual(
            data_sect.GetFileAddress() + 0x170000000, data_sect.GetLoadAddress(target)
        )
