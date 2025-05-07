"""
Test that the expression parser accounts for the underlying type of bitfield
enums when looking for matching values.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBitfieldEnum(TestBase):
    # Prior to clang-19, clang's DWARF v2 is missing missing DW_AT_type which
    # causes unsigned_max to appear as -1 instead of the "max" enumerator, whose
    # value is 3. From 19 onward, DW_AT_type is added as long as strict DWARF
    # is not enabled.
    @skipIf(dwarf_version=["<", "3"], compiler="clang", compiler_version=["<", "19.0"])
    def test_bitfield_enums(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr(
            "bfs",
            result_type="BitfieldStruct",
            result_children=[
                ValueCheck(name="signed_min", value="min"),
                ValueCheck(name="signed_other", value="-1"),
                ValueCheck(name="signed_max", value="max"),
                ValueCheck(name="unsigned_min", value="min"),
                ValueCheck(name="unsigned_other", value="1"),
                ValueCheck(name="unsigned_max", value="max"),
            ],
        )
