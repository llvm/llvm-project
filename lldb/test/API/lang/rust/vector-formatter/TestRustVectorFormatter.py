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

        self.expect(
            "frame variable vector",
            substrs=[
                "vector = size=3",
                "[0] = 1",
                "[1] = 2",
                "[2] = 3",
                "}",
            ],
        )

        self.expect(
            "expression vector",
            substrs=[
                "$",
                "size=3",
                "[0] = 1",
                "[1] = 2",
                "[2] = 3",
                "}",
            ],
        )
