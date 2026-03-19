"""
Test the memory commands operating on memory regions with holes (inaccessible
pages)
"""

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class MemoryHolesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// break here")

    def _prepare_inferior(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        # Avoid the expression evaluator, as it can allocate allocate memory
        # inside the holes we've deliberately left empty.
        self.memory = self.frame().FindVariable("mem_with_holes").GetValueAsUnsigned()
        self.pagesize = self.frame().FindVariable("pagesize").GetValueAsUnsigned()
        self.num_pages = (
            self.target().FindFirstGlobalVariable("num_pages").GetValueAsUnsigned()
        )
        positions = self.frame().FindVariable("positions")
        self.positions = [
            positions.GetChildAtIndex(i).GetValueAsUnsigned()
            for i in range(positions.GetNumChildren())
        ]
        self.assertEqual(len(self.positions), 5)

    def test_memory_read(self):
        self._prepare_inferior()

        error = lldb.SBError()
        content = self.process().ReadMemory(self.memory, 2 * self.pagesize, error)
        self.assertEqual(len(content), self.pagesize)
        self.assertEqual(content[0:7], b"needle\0")
        self.assertTrue(error.Fail())

    def test_memory_find(self):
        self._prepare_inferior()

        matches = [f"data found at location: {p:#x}" for p in self.positions]
        self.expect(
            f'memory find --count {len(self.positions)+1} --string "needle" '
            f"{self.memory:#x} {self.memory+self.pagesize*self.num_pages:#x}",
            substrs=matches + ["no more matches within the range"],
        )
