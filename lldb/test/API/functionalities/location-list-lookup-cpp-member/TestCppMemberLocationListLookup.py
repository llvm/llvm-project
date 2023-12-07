"""Test that lldb picks the correct DWARF location list entry with a return-pc out of bounds."""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CppMemberLocationListLookupTestCase(TestBase):
    def test(self):
        self.build()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.dbg.SetAsync(False)

        li = lldb.SBLaunchInfo(["a.out"])
        error = lldb.SBError()
        process = target.Launch(li, error)
        self.assertTrue(process.IsValid())
        self.assertTrue(process.is_stopped)

        # Find `bar` on the stack, then
        # find `this` local variable, then
        # check that we can read out the pointer value
        for f in process.GetSelectedThread().frames:
            if f.GetDisplayFunctionName().startswith("Foo::bar"):
                process.GetSelectedThread().SetSelectedFrame(f.idx)
                self.expect_expr("this", result_type="Foo *")
