import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftSymbolicExtendedExistential(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test symbolic extended existentials"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        if self.TraceOn():
            self.runCmd("v -d run -L s")
        frame = self.frame()
        var_s = frame.FindVariable("s")
        var_s_l0 = var_s.GetChildMemberWithName("l0")
        var_s_l1 = var_s.GetChildMemberWithName("l1")
        var_s_l2 = var_s.GetChildMemberWithName("l2")
        var_s_l3 = var_s.GetChildMemberWithName("l3")
        var_s_l4 = var_s.GetChildMemberWithName("l4")
        var_s_l5 = var_s.GetChildMemberWithName("l5")
        var_s_l6 = var_s.GetChildMemberWithName("l6")
        lldbutil.check_variable(self, var_s_l0, value="0")
        lldbutil.check_variable(self, var_s_l1, value="10")
        lldbutil.check_variable(self, var_s_l2, value="20")
        lldbutil.check_variable(self, var_s_l3, value="30")
        lldbutil.check_variable(self, var_s_l4, value="40")
        lldbutil.check_variable(self, var_s_l5, value="50")
        lldbutil.check_variable(self, var_s_l6, value="60")
        var_s_s1 = var_s.GetChildMemberWithName("s1")
        var_s_s2 = var_s.GetChildMemberWithName("s2")
        var_s_s3 = var_s.GetChildMemberWithName("s3")
        var_s_s4 = var_s.GetChildMemberWithName("s4")
        var_s_s5 = var_s.GetChildMemberWithName("s5")
        var_s_s6 = var_s.GetChildMemberWithName("s6")
        lldbutil.check_variable(self, var_s_s1, use_dynamic=True, summary="1...1")
        lldbutil.check_variable(self, var_s_s2, use_dynamic=True, summary="1...200")
        lldbutil.check_variable(self, var_s_s3, use_dynamic=True, summary="1...2")
        lldbutil.check_variable(self, var_s_s4, use_dynamic=True, summary="nil")
        lldbutil.check_variable(self, var_s_s5, use_dynamic=True, typename="a.C")
        # FIXME:
        # lldbutil.check_variable(self, var_s_s6, use_dynamic=True, summary="Int")

        self.expect("expression -- s.s1", substrs=['1...1'])
