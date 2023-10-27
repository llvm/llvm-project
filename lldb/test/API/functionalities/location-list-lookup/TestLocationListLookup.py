"""Test that lldb picks the correct DWARF location list entry with a return-pc out of bounds."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LocationListLookupTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @skipIf(oslist=["linux"], archs=["arm"])
    def test_loclist(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.dbg.SetAsync(False)

        li = lldb.SBLaunchInfo(["a.out"])
        error = lldb.SBError()
        process = target.Launch(li, error)
        self.assertTrue(process.IsValid())
        self.assertTrue(process.is_stopped)

        # Find `main` on the stack, then
        # find `argv` local variable, then
        # check that we can read the c-string in argv[0]
        for f in process.GetSelectedThread().frames:
            if f.GetDisplayFunctionName() == "main":
                argv = f.GetValueForVariablePath("argv").GetChildAtIndex(0)
                strm = lldb.SBStream()
                argv.GetDescription(strm)
                self.assertNotEqual(strm.GetData().find("a.out"), -1)
