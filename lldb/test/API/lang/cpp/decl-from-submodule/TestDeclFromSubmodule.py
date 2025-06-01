"""Test that decl lookup into submodules in C++ works as expected."""

import lldb
import shutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DeclFromSubmoduleTestCase(TestBase):
    # Requires DWARF debug info which is not retained when linking with link.exe.
    @skipIfWindows
    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))

        # FIXME: LLDB finds the decl for 'func' in the submodules correctly and hands it to Clang
        # but Sema rejects using the decl during name lookup because it is not marked "Visible".
        # However, this assertions still ensures that we at least don't fail to compile the
        # submodule (which would cause other errors to appear before the expression error, hence
        # we use "startstr").
        self.expect(
            "expr func(1, 2)",
            error=True,
            startstr="error: <user expression 0>:1:1: 'func' has unknown return type",
        )
