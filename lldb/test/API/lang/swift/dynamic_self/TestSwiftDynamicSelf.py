import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftDynamicSelf(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test_dynamic_self(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        frame = thread.frames[0]
        var_self = frame.FindVariable("self")
        self.assertEqual(var_self.GetNumChildren(), 0)
        dyn_self = var_self.GetDynamicValue(True)
        self.assertEqual(dyn_self.GetNumChildren(), 1)
        var_self_base = dyn_self.GetChildAtIndex(0)
        member_c = var_self_base.GetChildMemberWithName("c")
        member_v = var_self_base.GetChildMemberWithName("v")
        lldbutil.check_variable(self, member_c, False, value="100")
        lldbutil.check_variable(self, member_v, False, value="210")
