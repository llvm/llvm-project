import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftDynamicSelf(lldbtest.TestBase):

    def get_self_from_Base_method(self, frame):
        '''When stopped in a method in Base, get the 'self' with type Base.'''
        var_self = frame.FindVariable("self")
        self.assertEqual(var_self.GetNumChildren(), 2)
        return var_self

    def get_self_as_Base_from_Child_method(self, frame):
        '''When stopped in a method in Child, get the 'self' with type Base.'''
        var_self = frame.FindVariable("self")
        dyn_self = var_self.GetDynamicValue(True)
        self.assertEqual(dyn_self.GetNumChildren(), 1)
        var_self_base = dyn_self.GetChildAtIndex(0)
        self.assertEqual(var_self_base.GetNumChildren(), 2)
        return var_self_base

    def check_members(self, var_self_base, c_val, v_val):
        member_c = var_self_base.GetChildMemberWithName("c")
        member_v = var_self_base.GetChildMemberWithName("v")
        lldbutil.check_variable(self, member_c, False, value=c_val)
        lldbutil.check_variable(self, member_v, False, value=v_val)

    @swiftTest
    def test_dynamic_self(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # In Base.init.
        frame = thread.frames[0]
        self.check_members(self.get_self_from_Base_method(frame), "100", "200")

        lldbutil.continue_to_breakpoint(process, bkpt) # Stop in Child.init.
        frame = thread.frames[0]
        # When stopped in Child.init(), the first child of 'self' is 'a.Base'.
        self.assertEqual(frame.FindVariable("self").GetNumChildren(), 1)
        self.check_members(self.get_self_as_Base_from_Child_method(frame),
                "100", "210")

        lldbutil.continue_to_breakpoint(process, bkpt) # Stop in Child.show.
        frame = thread.frames[0]
        # When stopped in Child.show(), 'self' doesn't have a child.
        self.assertEqual(frame.FindVariable("self").GetNumChildren(), 0)
        self.check_members(self.get_self_as_Base_from_Child_method(frame),
                "100", "220")
