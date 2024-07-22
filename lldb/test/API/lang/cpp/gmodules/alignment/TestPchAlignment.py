"""
Tests that we correctly track AST layout info
(specifically alignment) when moving AST nodes
between ClangASTImporter instances (in this case,
from pch to executable to expression AST).
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPchAlignment(TestBase):
    @add_test_categories(["gmodules"])
    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "return data", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable data",
            substrs=["row = 1", "col = 2", "row = 3", "col = 4", "stride = 5"],
        )

    @add_test_categories(["gmodules"])
    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "return data", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "data",
            result_type="MatrixData",
            result_children=[
                ValueCheck(
                    name="section",
                    children=[
                        ValueCheck(
                            name="origin",
                            children=[
                                ValueCheck(name="row", value="1"),
                                ValueCheck(name="col", value="2"),
                            ],
                        ),
                        ValueCheck(
                            name="size",
                            children=[
                                ValueCheck(name="row", value="3"),
                                ValueCheck(name="col", value="4"),
                            ],
                        ),
                    ],
                ),
                ValueCheck(name="stride", value="5"),
            ],
        )
