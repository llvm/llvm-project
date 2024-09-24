import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


@skipIf(oslist=['linux'])
class TestSwiftAnonymousClangTypes(lldbtest.TestBase):
    @swiftTest
    def test(self):
        self.build()

        target, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        def check(field, value):
            self.assertTrue(field.IsValid())
            lldbutil.check_variable(self, field, False, value=value)       

        twoStructs = target.FindFirstGlobalVariable("twoStructs")
        self.assertTrue(twoStructs.IsValid())

        field0 = twoStructs.GetChildAtIndex(0)
        self.assertTrue(field0.IsValid())
        check(field0.GetChildMemberWithName("x"), value="1")
        check(field0.GetChildMemberWithName("y"), value="2")
        check(field0.GetChildMemberWithName("z"), value="3")

        field1 = twoStructs.GetChildAtIndex(1)
        self.assertTrue(field1.IsValid())
        check(field1.GetChildMemberWithName("a"), value="4")


        twoUnions = target.FindFirstGlobalVariable("twoUnions")
        self.assertTrue(twoUnions.IsValid())
        field0 = twoUnions.GetChildAtIndex(0)
        self.assertTrue(field0.IsValid())
        field0_0 = field0.GetChildAtIndex(0)
        check(field0_0.GetChildMemberWithName("x"), value="2")
        field0_1 = field0.GetChildAtIndex(1)
        check(field0_1.GetChildMemberWithName("y"), value="2")
        check(field0_1.GetChildMemberWithName("z"), value="3")

        field1 = twoUnions.GetChildAtIndex(1)
        self.assertTrue(field1.IsValid())
        field1_0 = field1.GetChildAtIndex(0)
        check(field1_0.GetChildMemberWithName("a"), value="4")
        check(field1_0.GetChildMemberWithName("b"), value="5")
        check(field1_0.GetChildMemberWithName("c"), value="6")
        field1_1 = field1.GetChildAtIndex(1)
        check(field1_1.GetChildMemberWithName("d"), value="4")
        check(field1_1.GetChildMemberWithName("e"), value="5")
