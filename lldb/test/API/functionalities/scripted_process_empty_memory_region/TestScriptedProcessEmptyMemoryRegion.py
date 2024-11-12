"""
Test python scripted process which returns an empty SBMemoryRegionInfo
"""

import os, shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class ScriptedProcessEmptyMemoryRegion(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_scripted_process_empty_memory_region(self):
        """Test that lldb handles an empty SBMemoryRegionInfo object from
        a scripted process plugin."""
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        scripted_process_example_relpath = "dummy_scripted_process.py"
        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), scripted_process_example_relpath)
        )

        self.expect("memory region 0", error=True, substrs=["Invalid memory region"])

        self.expect("expr -- 5", substrs=["5"])
