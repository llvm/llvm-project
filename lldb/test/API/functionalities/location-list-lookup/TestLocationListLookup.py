"""Test that lldb picks the correct DWARF location list entry with a return-pc out of bounds."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LocationListLookupTestCase(TestBase):
    @skipIf(oslist=["linux"], archs=["arm"])
    def test_loclist(self):
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
        # make sure we can read out the local
        # variables (with both `frame var` and `expr`)
        for f in process.GetSelectedThread().frames:
            if f.GetDisplayFunctionName().startswith("Foo::bar"):
                argv = f.GetValueForVariablePath("argv").GetChildAtIndex(0)
                strm = lldb.SBStream()
                argv.GetDescription(strm)
                self.assertNotEqual(strm.GetData().find("a.out"), -1)

                process.GetSelectedThread().SetSelectedFrame(f.idx)
                self.expect_expr("this", result_type="Foo *")
