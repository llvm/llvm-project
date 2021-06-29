"""
Test errors from 'memory tag' commands on unsupported platforms.
Tests for the only supported platform, AArch64 Linux, are in
API/linux/aarch64/.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class MemoryTagTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_memory_tag_read_unsupported(self):
        """Test that "memory tag read" errors on unsupported platforms"""
        if not self.isAArch64MTE():
            self.skipTest("Requires a target without AArch64 MTE.")

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp",
                            line_number('main.cpp', '// Breakpoint here'),
                                        num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("memory tag read 0 1",
                    substrs=["error: This architecture does not support memory tagging"],
                    error=True)
