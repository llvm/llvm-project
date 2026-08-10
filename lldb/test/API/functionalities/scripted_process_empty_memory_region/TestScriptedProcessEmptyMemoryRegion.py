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

    # imports examples/python/templates/scripted_process.py
    # which only has register definitions for x86_64 and arm64.
    @skipIf(archs=no_match(["arm64", "x86_64"]))
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
