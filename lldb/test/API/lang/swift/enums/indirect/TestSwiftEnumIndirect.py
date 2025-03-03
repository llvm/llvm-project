import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestCase(TestBase):
    @swiftTest
    def test(self):
        """Test indirect enums"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        if self.TraceOn():
            self.expect('v')
        frame = thread.frames[0]
        generic_s = frame.FindVariable("generic_s")
        s = generic_s.GetChildMemberWithName("s")
        t = s.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")

        generic_large_s = frame.FindVariable("generic_large_s")
        s = generic_s.GetChildMemberWithName("s")
        t = s.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")

        generic_c = frame.FindVariable("generic_c")
        c = generic_c.GetChildMemberWithName("c")
        t = c.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")

        multi_s = frame.FindVariable("multi_s")
        s = multi_s.GetChildMemberWithName("s")
        t = s.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")

        multi_c = frame.FindVariable("multi_c")
        lldbutil.check_variable(self, multi_c, False, value="c")

        tuple_a = frame.FindVariable("tuple_a")
        a = tuple_a.GetChildMemberWithName("a")
        self.assertEqual(a.GetNumChildren(), 2)
        lldbutil.check_variable(self, a.GetChildAtIndex(0), False, value="23")
        lldbutil.check_variable(self, a.GetChildAtIndex(1), False, summary='"hello"')
        tuple_b = frame.FindVariable("tuple_b")
        b = tuple_b.GetChildMemberWithName("b")
        self.assertEqual(b.GetNumChildren(), 1)
        lldbutil.check_variable(self, b.GetChildAtIndex(0), False, value="42")
        tuple_c = frame.FindVariable("tuple_c")
        c = tuple_c.GetChildMemberWithName("c")
        lldbutil.check_variable(self, c, False, value="32")
        tuple_d = frame.FindVariable("tuple_d")
        lldbutil.check_variable(self, tuple_d, False, value="d")

        tree = frame.FindVariable("tree")
        n0 = tree.GetChildMemberWithName("node")
        n1 = n0.GetChildAtIndex(0)
        n2 = n1.GetChildAtIndex(0)
        l0 = n2.GetChildAtIndex(0)
        lldbutil.check_variable(self, l0.GetChildAtIndex(0), False, value="1")
        l1 = n2.GetChildAtIndex(1)
        lldbutil.check_variable(self, l1.GetChildAtIndex(0), False, value="2")
        l2 = n0.GetChildAtIndex(1)
        lldbutil.check_variable(self, l2.GetChildAtIndex(0), False, value="3")
