"""
Test diamond inheritance.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_with_sbvalue(self):
        """
        Test that virtual base classes work in when SBValue objects are
        used to explore the class.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        j1 = self.frame().FindVariable("j1")
        j1_Derived1 = j1.GetChildAtIndex(0)
        j1_Derived2 = j1.GetChildAtIndex(1)
        j1_Derived1_VBase = j1_Derived1.GetChildAtIndex(0)
        j1_Derived2_VBase = j1_Derived2.GetChildAtIndex(0)
        j1_Derived1_VBase_m_value = j1_Derived1_VBase.GetChildAtIndex(0)
        j1_Derived2_VBase_m_value = j1_Derived2_VBase.GetChildAtIndex(0)

        self.assertEqual(
            j1_Derived1_VBase.GetLoadAddress(),
            j1_Derived2_VBase.GetLoadAddress(),
            "ensure virtual base class is the same between Derived1 and Derived2",
        )
        self.assertEqual(
            j1_Derived1_VBase_m_value.GetValueAsUnsigned(1),
            j1_Derived2_VBase_m_value.GetValueAsUnsigned(2),
            "ensure m_value in VBase is the same",
        )
        self.assertEqual(
            self.frame()
            .FindVariable("d")
            .GetChildAtIndex(0)
            .GetChildAtIndex(0)
            .GetValueAsUnsigned(0),
            12345,
            "ensure Derived2 from j1 is correct",
        )

        # This reassigns 'd' to point to 'j2'.
        self.thread().StepOver()

        self.assertEqual(
            self.frame()
            .FindVariable("d")
            .GetChildAtIndex(0)
            .GetChildAtIndex(0)
            .GetValueAsUnsigned(0),
            12346,
            "ensure Derived2 from j2 is correct",
        )

    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.cpp")
        )

        # All the children of j1.
        children = [
            ValueCheck(
                type="Derived1",
                children=[
                    ValueCheck(
                        type="VBase",
                        children=[
                            ValueCheck(type="int", name="m_value", value="12345")
                        ],
                    )
                ],
            ),
            ValueCheck(
                type="Derived2",
                children=[
                    ValueCheck(
                        type="VBase",
                        children=[
                            ValueCheck(type="int", name="m_value", value="12345")
                        ],
                    )
                ],
            ),
            ValueCheck(type="long", value="1"),
        ]
        # Try using the class with expression evaluator/variable paths.
        self.expect_expr("j1", result_type="Joiner1", result_children=children)
        self.expect_var_path("j1", type="Joiner1", children=children)

        # Use the expression evaluator to access the members.
        self.expect_expr("j1.x", result_type="long", result_value="1")
        self.expect_expr("j1.m_value", result_type="int", result_value="12345")

        # Use variable paths to access the members.
        self.expect_var_path("j1.x", type="long", value="1")

    @expectedFailureAll
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.cpp")
        )
        # FIXME: This is completely broken and 'succeeds' with an error that
        # there is noch such value/member in Joiner1. Move this up to the test
        # above when fixed.
        self.expect_var_path("j1.m_value", type="int", value="12345")
