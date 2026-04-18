"""Test __ptrauth type qualifier."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrAuth(TestBase):
    @skipUnlessArm64eSupported
    def test(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break in main", lldb.SBFileSpec("main.c")
        )

        self.expect(
            "target var valid0",
            substrs=["(int *__ptrauth(2,0,0))", "valid0 =", "(actual=0x"],
        )
        self.expect(
            "target var valid1",
            substrs=["(int *__ptrauth(2,0,0) *)", "valid1 ="],
        )
        self.expect(
            "target var valid2",
            substrs=["(__ptrauth(2,0,0) intp)", "valid2 =", "(actual=0x"],
        )
        self.expect(
            "target var valid3",
            substrs=["(__ptrauth(2,0,0) intp *)", "valid3 ="],
        )
        self.expect(
            "target var valid4",
            substrs=["(__ptrauth(2,0,0) intp)", "valid4 =", "(actual=0x"],
        )

        bkpt = self.target().BreakpointCreateBySourceRegex(
            "// break in test_code", lldb.SBFileSpec("main.c")
        )
        self.assertGreater(bkpt.GetNumLocations(), 0)
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect(
            "fr var pSpecial",
            substrs=["(__ptrauth(2,0,0) intp)", "pSpecial =", "(actual=0x"],
        )
        self.expect("fr var *pSpecial", substrs=["5"])
        self.expect(
            "p pSpecial",
            substrs=["(__ptrauth(2,0,0) intp)", "(actual=0x"],
        )
