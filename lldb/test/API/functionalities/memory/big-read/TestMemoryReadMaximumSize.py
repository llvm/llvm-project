"""
Test the maximum memory read setting.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestMemoryReadMaximumSize(TestBase):
    def test_memory_read_max_setting(self):
        """Test the target.max-memory-read-size setting."""
        self.build()
        (
            self.target,
            self.process,
            self.thread,
            self.bp,
        ) = lldbutil.run_to_source_breakpoint(
            self, "breakpoint here", lldb.SBFileSpec("main.c")
        )
        self.assertTrue(self.bp.IsValid())

        self.runCmd("settings set target.max-memory-read-size 1024")

        self.expect(
            "mem rea -f x -s 4 -c 2048 `&c`",
            error=True,
            substrs=["Normally, 'memory read' will not read over 1024 bytes of data"],
        )
        self.runCmd("settings set target.max-memory-read-size `2048 * sizeof(int)`")
        self.expect("mem rea -f x -s 4 -c 2048 `&c`", substrs=["feed"])
