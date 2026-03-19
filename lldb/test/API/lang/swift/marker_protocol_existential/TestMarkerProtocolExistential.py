import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestMarkerProtocolExistential(lldbtest.TestBase):
    @swiftTest
    def test_marker_only(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break marker only", lldb.SBFileSpec("main.swift")
        )

        s = self.frame().FindVariable("v")
        self.assertEqual(s.GetTypeName(), "a.S")
        x = s.GetChildMemberWithName("x")
        lldbutil.check_variable(self, x, value="42")

    @swiftTest
    def test_marker_composition(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break composition", lldb.SBFileSpec("main.swift")
        )

        t = self.frame().FindVariable("v")
        a = t.GetChildMemberWithName("a")
        lldbutil.check_variable(self, a, value="10")

    @swiftTest
    def test_two_markers(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break two markers", lldb.SBFileSpec("main.swift")
        )

        u = self.frame().FindVariable("v")
        self.assertEqual(u.GetTypeName(), "a.U")
        b = u.GetChildMemberWithName("b")
        lldbutil.check_variable(self, b, value="20")

    @swiftTest
    def test_any_and_marker(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break any and marker", lldb.SBFileSpec("main.swift")
        )

        s = self.frame().FindVariable("v")
        self.assertEqual(s.GetTypeName(), "a.S")
        x = s.GetChildMemberWithName("x")
        lldbutil.check_variable(self, x, value="42")

    @swiftTest
    def test_any_marker_and_non_marker(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break any marker and non marker", lldb.SBFileSpec("main.swift")
        )

        v = self.frame().FindVariable("v")
        d = v.GetChildMemberWithName("d")
        lldbutil.check_variable(self, d, value="30")
