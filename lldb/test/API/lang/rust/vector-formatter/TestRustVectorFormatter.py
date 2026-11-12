"""
Make sure Rust vector data formatter works.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestRustVectorFormatter(TestBase):
    def test_vec_formatter(self):
        core_yaml_path = self.getSourcePath("main.core.yaml")
        main_yaml_path = self.getSourcePath("main.yaml")
        self.yaml2obj(core_yaml_path, self.getBuildArtifact("main.core"))
        self.yaml2obj(main_yaml_path, self.getBuildArtifact("main"))
        target = self.dbg.CreateTarget(self.getBuildArtifact("main"))
        process = target.LoadCore(self.getBuildArtifact("main.core"))
        self.assertTrue(process.IsValid())

        self.expect_var_path(
            "vector",
            summary="size=3",
            children=[
                ValueCheck(name="[0]", value="1"),
                ValueCheck(name="[1]", value="2"),
                ValueCheck(name="[2]", value="3"),
            ],
        )

        self.expect_expr(
            "vector",
            result_summary="size=3",
            result_children=[
                ValueCheck(name="[0]", value="1"),
                ValueCheck(name="[1]", value="2"),
                ValueCheck(name="[2]", value="3"),
            ],
        )
