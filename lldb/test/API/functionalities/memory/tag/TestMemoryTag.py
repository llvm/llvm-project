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
    NO_DEBUG_INFO_TESTCASE = True

    def test_memory_tag_read_unsupported(self):
        """Test that "memory tag read" errors on unsupported platforms"""
        if self.isAArch64MTE():
            self.skipTest("Requires a target without AArch64 MTE.")

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            line_number("main.cpp", "// Breakpoint here"),
            num_expected_locations=1,
        )
        self.runCmd("run", RUN_SUCCEEDED)

        # If you're on AArch64 you could have MTE but the remote process
        # must also support it. If you're on any other architecture you
        # won't have any tagging at all. So the error message is different.
        if self.isAArch64():
            expected = "error: Process does not support memory tagging"
        else:
            expected = "error: This architecture does not support memory tagging"

        self.expect("memory tag read 0 1", substrs=[expected], error=True)
        self.expect("memory tag write 0 1 2", substrs=[expected], error=True)
