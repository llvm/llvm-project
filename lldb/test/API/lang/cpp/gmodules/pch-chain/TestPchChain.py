"""
Tests that we correctly track AST layout info
(specifically alignment) when moving AST nodes
between several ClangASTImporter instances
(in this case, from a pch chain to executable
to expression AST).
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPchChain(TestBase):
    @add_test_categories(["gmodules"])
    @expectedFailureAll("Chained pch debugging currently not fully supported")
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", 9, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "frame variable data",
            substrs=["row = 1", "col = 2", "row = 3", "col = 4", "stride = 5"],
        )

    @add_test_categories(["gmodules"])
    @expectedFailureAll("Chained pch debugging currently not fully supported")
    def test_frame_var(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", 9, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

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
