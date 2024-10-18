import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftBigMultiPayloadEnum(TestBase):
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        a1 = self.frame().FindVariable("a1")
        lldbutil.check_variable(self, a1, False, typename="a.Request", value="a1")

        a2 = self.frame().FindVariable("a2")
        lldbutil.check_variable(self, a2, False, typename="a.Request", value="a2")

        a3 = self.frame().FindVariable("a3")
        lldbutil.check_variable(self, a3, False, typename="a.Request", value="a3")

        a4 = self.frame().FindVariable("a4")
        lldbutil.check_variable(self, a4, False, typename="a.Request", value="a4")

        a5 = self.frame().FindVariable("a5")
        lldbutil.check_variable(self, a5, False, typename="a.Request", value="a5")

        a6_item1 = self.frame().FindVariable("a6_item1")
        lldbutil.check_variable(self, a6_item1, False, typename="a.Request", value="a6")

        item1 = a6_item1.GetChildAtIndex(0)
        lldbutil.check_variable(self, item1, False, typename="a.A6", value="item1")

        a6_item2 = self.frame().FindVariable("a6_item2")
        lldbutil.check_variable(self, a6_item2, False, typename="a.Request", value="a6")

        item2 = a6_item2.GetChildAtIndex(0)
        lldbutil.check_variable(self, item2, False, typename="a.A6", value="item2")

        a7_item1 = self.frame().FindVariable("a7_item1")
        lldbutil.check_variable(self, a7_item1, False, typename="a.Request", value="a7")

        item1 = a7_item1.GetChildAtIndex(0)
        lldbutil.check_variable(self, item1, False, typename="a.A7", value="item1")

        a7_item2 = self.frame().FindVariable("a7_item2")
        lldbutil.check_variable(self, a7_item2, False, typename="a.Request", value="a7")

        item2 = a7_item2.GetChildAtIndex(0)
        lldbutil.check_variable(self, item2, False, typename="a.A7", value="item2")
