"""Test that importing modules in Objective-C works as expected."""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCModulesAutoImportTestCase(TestBase):

    @skipIf(macos_version=["<", "10.12"])
    @skipIf(compiler="clang", compiler_version=["<", "19.0"])
    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m", False)
        )

        self.runCmd("settings set target.auto-import-clang-modules true")
        self.expect_expr("getpid()", result_type="pid_t")
