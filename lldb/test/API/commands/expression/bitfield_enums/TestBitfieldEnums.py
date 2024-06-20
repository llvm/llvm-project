"""
Test that the expression parser accounts for the underlying type of bitfield
enums when looking for matching values.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBitfieldEnum(TestBase):
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
