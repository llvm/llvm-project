"""
Tests C99's flexible array members.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                lldb.SBFileSpec("main.c"))

        self.expect_var_path("c->flexible", type="char[]", summary='"contents"')
        # self.expect_var_path("sc->flexible", type="signed char[]", summary='"contents"')
        self.expect_var_path("uc->flexible", type="unsigned char[]", summary='"contents"')
        # TODO: Make this work
        self.expect("expr c->flexible", error=True,
                substrs=["incomplete", "char[]"])
        self.expect("expr sc->flexible", error=True,
                substrs=["incomplete", "signed char[]"])
        self.expect("expr uc->flexible", error=True,
                substrs=["incomplete", "unsigned char[]"])
