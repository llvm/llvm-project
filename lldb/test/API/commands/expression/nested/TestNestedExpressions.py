"""
Test calling an expression with errors that a FixIt can fix.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NestedExpressions(TestBase):
    def test_enum_in_nested_structs(self):
        """
        Test expressions that references an enumeration in nested structs.
        """
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target, "Target: %s is not valid." % (exe_path))
        self.expect_expr(
            "A::B::C::EnumType::Eleven",
            result_type="A::B::C::EnumType",
            result_value="Eleven",
        )

    def test_struct_in_nested_structs(self):
        """
        Test expressions that references a struct in nested structs.
        """
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target, "Target: %s is not valid." % (exe_path))
        self.expect_expr("sizeof(A::B::C)", result_value="1")
        self.expect_expr("sizeof(A::B)", result_value="2")

    # Fails on Windows for unknown reasons.
    @skipIfWindows
    def test_static_in_nested_structs(self):
        """
        Test expressions that references a static variable in nested structs.
        """
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to evaluate expressions", lldb.SBFileSpec("main.cpp")
        )
        self.expect_expr(
            "A::B::C::enum_static",
            result_type="A::B::C::EnumType",
            result_value="Eleven",
        )

    def test_enum_in_nested_namespaces(self):
        """
        Test expressions that references an enumeration in nested namespaces.
        """
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target, "Target: %s is not valid." % (exe_path))
        self.expect_expr(
            "a::b::c::Color::Blue", result_type="a::b::c::Color", result_value="Blue"
        )

    def test_static_in_nested_namespaces(self):
        """
        Test expressions that references an enumeration in nested namespaces.
        """
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to evaluate expressions", lldb.SBFileSpec("main.cpp")
        )
        self.expect_expr("a::b::c::d", result_type="int", result_value="12")
