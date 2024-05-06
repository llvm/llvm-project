import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftGenericFunction(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test display of generic function names"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'f')

        # Live process:
        stream = lldb.SBStream()
        self.frame().GetDescription(stream)
        desc = stream.GetData()
        # It's debatable whether C's generic parameter should be displayed here.
        self.assertIn("C.f<Int>(t=1, u=2)", desc)

        # Dead process + debug info:
        process.Kill()
        stream = lldb.SBStream()
        bkpt.GetLocationAtIndex(0).GetDescription(stream, 1)
        desc = stream.GetData()
        self.assertIn("C.f<T>(T, U) -> ()", desc)

        # Demangling only:
        fs = target.FindFunctions("f")
        self.assertTrue(fs)
        desc = fs[0].GetFunction().GetDisplayName()
        self.assertIn("C.f<τ_0_0>(_:_:)", desc)

