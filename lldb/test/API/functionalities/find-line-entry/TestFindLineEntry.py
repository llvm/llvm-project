"""
Test that SBCompileUnit::FindLineEntryIndex works correctly.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FindLineEntry(TestBase):
    def test_compile_unit_find_line_entry_index(self):
        """Test the CompileUnit LineEntryIndex lookup API"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target.IsValid(), "Target is not valid")

        self.file = lldb.SBFileSpec("main.c")
        sc_list = self.target.FindCompileUnits(self.file)
        self.assertEqual(len(sc_list), 1)
        cu = sc_list[0].GetCompileUnit()
        self.assertTrue(cu.IsValid(), "CompileUnit is not valid")

        # First look for valid line
        self.line = line_number("main.c", "int change_me")
        self.assertNotEqual(
            cu.FindLineEntryIndex(0, self.line, self.file),
            lldb.LLDB_INVALID_LINE_NUMBER,
        )

        # Then look for a line out of bound
        self.assertEqual(
            cu.FindLineEntryIndex(0, 42, self.file), lldb.LLDB_INVALID_LINE_NUMBER
        )
