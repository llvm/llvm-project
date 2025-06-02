import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        """Test indirect enums"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        if self.TraceOn():
            self.expect('v -d run -T')
        generic_s = frame.FindVariable("generic_s").GetSyntheticValue()
        t = generic_s.GetChildMemberWithName("t")
        print(t)
        lldbutil.check_variable(self, t, False, value="123")
 
        generic_large_s = frame.FindVariable("generic_large_s").GetSyntheticValue()
        t = generic_s.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")
 
        generic_c = frame.FindVariable("generic_c").GetSyntheticValue()
        t = generic_c.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")
 
        multi_s = frame.FindVariable("multi_s").GetSyntheticValue()
        t = multi_s.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")
 
        multi_c = frame.FindVariable("multi_c").GetSyntheticValue()
        t = multi_c.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value="123")
 
        tuple_a = frame.FindVariable("tuple_a").GetSyntheticValue()
        lldbutil.check_variable(self, tuple_a, False, value="a")
        self.assertEqual(tuple_a.GetNumChildren(), 2)
        lldbutil.check_variable(self, tuple_a.GetChildAtIndex(0), False, value="23")
        lldbutil.check_variable(self, tuple_a.GetChildAtIndex(1), False, summary='"hello"')
        tuple_b = frame.FindVariable("tuple_b").GetSyntheticValue()
        lldbutil.check_variable(self, tuple_b, False, value="b")
        self.assertEqual(tuple_b.GetNumChildren(), 1)
        lldbutil.check_variable(self, tuple_b.GetChildAtIndex(0), False, value="42")
        tuple_c = frame.FindVariable("tuple_c").GetSyntheticValue()
        lldbutil.check_variable(self, tuple_c, False, value="c")
        self.assertEqual(tuple_c.GetNumChildren(), 1)
        lldbutil.check_variable(self, tuple_c.GetChildAtIndex(0), False, value="32")
        tuple_d = frame.FindVariable("tuple_d").GetSyntheticValue()
        lldbutil.check_variable(self, tuple_d, False, value="d")
        self.assertEqual(tuple_d.GetNumChildren(), 1)
        lldbutil.check_variable(self, tuple_d.GetChildAtIndex(0), False, value="16")
        tuple_e = frame.FindVariable("tuple_e")
        lldbutil.check_variable(self, tuple_e, False, value="e")
        self.assertEqual(tuple_e.GetNumChildren(), 0)

        tree = frame.FindVariable("tree").GetSyntheticValue()
        node = tree.GetChildAtIndex(0)
        leaf1 = node.GetChildAtIndex(0)
        leaf2 = node.GetChildAtIndex(1)
        lldbutil.check_variable(self, leaf1, False, value="leaf")
        lldbutil.check_variable(self, leaf1.GetChildAtIndex(0), False, value="1")
        lldbutil.check_variable(self, leaf2, False, value="leaf")
        lldbutil.check_variable(self, leaf2.GetChildAtIndex(0), False, value="2")
        leaf3 = tree.GetChildAtIndex(1)
        lldbutil.check_variable(self, leaf3, False, value="leaf")
        lldbutil.check_variable(self, leaf3.GetChildAtIndex(0), False, value="3")

