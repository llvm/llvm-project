"""Test that hidden ivars in a shared library are visible from the main executable."""

import unittest

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfTargetDoesNotSupportSharedLibraries()
@skipIf(archs=["i386"])  # requires modern objc runtime
class HiddenIvarsTestCase(TestBase):
    SHARED_BUILD_TESTCASE = True

    def common_setup(self, strip):
        exe_name = "stripped/a.out" if strip else "a.out"
        return lldbutil.run_to_source_breakpoint(
            self,
            "// breakpoint1",
            lldb.SBFileSpec("main.m"),
            exe_name=exe_name,
            extra_images=["InternalDefiner"],
        )

    @skipIf(
        debug_info=no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM",
    )
    def test_expr_stripped(self):
        self.build()
        self.expr(strip=True)

    def test_expr(self):
        self.build()
        self.expr(strip=False)

    @skipIf(
        debug_info=no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM",
    )
    def test_frame_variable_stripped(self):
        self.build()
        self.frame_var(strip=True)

    def test_frame_variable(self):
        self.build()
        self.frame_var(strip=False)

    @unittest.expectedFailure  # rdar://18683637
    def test_frame_variable_across_modules(self):
        self.build()
        self.common_setup(False)
        self.expect_var_path("k->bar", value="3")

    def expr(self, strip):
        self.common_setup(strip)

        self.expect_expr("j->_definer->foo", result_value="4")
        self.expect_expr("j->_definer->bar", result_value="5")

        self.expect_expr(
            "*(j->_definer)",
            result_type="InternalDefiner",
            result_children=[
                ValueCheck(name="NSObject"),
                ValueCheck(name="foo", value="4"),
                ValueCheck(name="bar", value="5"),
            ],
        )

        self.expect_expr("k->foo", result_value="2")
        self.expect_expr("k->bar", result_value="3")

        self.expect_expr("k.filteredDataSource", result_summary='@"2 elements"')

        self.expect_expr(
            "*k",
            result_type="InheritContainer",
            result_children=[
                ValueCheck(
                    name="InternalDefiner",
                    children=[
                        ValueCheck(name="NSObject"),
                        ValueCheck(name="foo", value="2"),
                        ValueCheck(name="bar", value="3"),
                    ],
                ),
                ValueCheck(name="_filteredDataSource", summary='@"2 elements"'),
            ],
        )

    def frame_var(self, strip):
        self.common_setup(strip)

        self.expect_var_path("j->_definer->foo", value="4")
        self.expect_var_path("j->_definer->bar", value="5")

        self.expect_var_path(
            "*j->_definer",
            children=[
                ValueCheck(name="NSObject"),
                ValueCheck(name="foo", value="4"),
                ValueCheck(name="bar", value="5"),
            ],
        )

        self.expect_var_path("k->foo", value="2")
        self.expect_var_path("k->_filteredDataSource", summary='@"2 elements"')

        self.expect_var_path(
            "*k",
            type="InheritContainer",
            children=[
                ValueCheck(
                    name="InternalDefiner",
                    children=[
                        ValueCheck(name="NSObject"),
                        ValueCheck(name="foo", value="2"),
                        ValueCheck(name="bar", value="3"),
                    ],
                ),
                ValueCheck(name="_filteredDataSource", summary='@"2 elements"'),
            ],
        )
