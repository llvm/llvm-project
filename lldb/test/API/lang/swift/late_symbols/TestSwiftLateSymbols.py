import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import shutil
import os

class TestSwiftLateSymbols(TestBase):
    @swiftTest
    @skipUnlessDarwin
    @skipIf(debug_info=no_match(["dsym"]))
    def test_any_object_type(self):
        """Test the AnyObject type"""
        self.build()
        dsym = self.getBuildArtifact('a.out.dSYM')
        stash = self.getBuildArtifact('hidden.noindex')
        os.unlink(self.getBuildArtifact('main.swift.o'))
        os.makedirs(stash)
        shutil.move(dsym, stash)
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'breakpoint')
        # return to main(), at a place where all variables are available
        thread.StepOut()

        frame = thread.frames[0]
        var_object = frame.FindVariable("object", lldb.eNoDynamicValues)
        self.assertFalse(var_object.IsValid())

        self.expect('add-dsym ' + stash + '/a.out.dSYM')
        frame = thread.frames[0]
        var_object = frame.FindVariable("object", lldb.eNoDynamicValues)
        self.assertTrue(var_object.IsValid())

        lldbutil.check_variable(
            self,
            var_object,
            use_dynamic=False,
            typename="bridging.h.FromC")
        var_object_x = var_object.GetDynamicValue(
            lldb.eDynamicCanRunTarget).GetChildMemberWithName("i")
        lldbutil.check_variable(
            self,
            var_object_x,
            use_dynamic=False,
            value='23',
            typename="Swift.Int32")
