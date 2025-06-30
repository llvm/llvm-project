"""Test that decl lookup into submodules in C++ works as expected."""

import lldb
import shutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DeclFromSubmoduleTestCase(TestBase):
    # Requires DWARF debug info which is not retained when linking with link.exe.
    @skipIfWindows
    # Lookup for decls in submodules fails in Linux
    @expectedFailureAll(oslist=["linux"])
    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("func(1, 2)", result_type="int", result_value="3")
        self.expect_expr("func(1)", result_type="int", result_value="1")
