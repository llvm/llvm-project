"""
Test that we can print and call closures passed in various contexts
"""

import os
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


def check_not_captured_error(test, frame, var_name, parent_function):
    expected_error = (
        f"A variable named '{var_name}' existed in function '{parent_function}'"
    )
    value = frame.EvaluateExpression(var_name)
    error = value.GetError().GetCString()
    test.assertIn(expected_error, error)

    value = frame.EvaluateExpression(f"1 + {var_name} + 1")
    error = value.GetError().GetCString()
    test.assertIn(expected_error, error)

    test.expect(f"frame variable {var_name}", substrs=[expected_error], error=True)


def check_no_enhanced_diagnostic(test, frame, var_name):
    forbidden_str = "A variable named"
    value = frame.EvaluateExpression(var_name)
    error = value.GetError().GetCString()
    test.assertNotIn(forbidden_str, error)

    value = frame.EvaluateExpression(f"1 + {var_name} + 1")
    error = value.GetError().GetCString()
    test.assertNotIn(forbidden_str, error)

    test.expect(
        f"frame variable {var_name}",
        substrs=[forbidden_str],
        matching=False,
        error=True,
    )


class TestSwiftClosureVarNotCaptured(TestBase):
    def get_to_bkpt(self, bkpt_name):
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, bkpt_name, lldb.SBFileSpec("main.swift")
        )
        target.BreakpointDelete(bkpt.GetID())
        return (target, process, thread)

    @swiftTest
    def test_simple_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_simple_closure")
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_1(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_1(arg:)")
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    def test_nested_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_double_closure_1")
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_2(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_2(arg:)")
        check_not_captured_error(
            self, thread.frames[0], "var_in_outer_closure", "closure #1 in func_2(arg:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

        lldbutil.continue_to_source_breakpoint(
            self, process, "break_double_closure_2", lldb.SBFileSpec("main.swift")
        )
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_2(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_2(arg:)")
        check_not_captured_error(
            self, thread.frames[0], "var_in_outer_closure", "closure #1 in func_2(arg:)"
        )
        check_not_captured_error(
            self, thread.frames[0], "shadowed_var", "closure #1 in func_2(arg:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    # Async variable inspection on Linux/Windows are still problematic.
    @skipIf(oslist=["windows", "linux"])
    def test_async_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_async_closure_1")
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_3(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_3(arg:)")
        check_not_captured_error(
            self, thread.frames[0], "var_in_outer_closure", "closure #1 in func_3(arg:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

        lldbutil.continue_to_source_breakpoint(
            self, process, "break_async_closure_2", lldb.SBFileSpec("main.swift")
        )
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_3(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_3(arg:)")
        check_not_captured_error(
            self, thread.frames[0], "var_in_outer_closure", "closure #1 in func_3(arg:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    def test_ctor_class_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_ctor_class")
        check_not_captured_error(
            self, thread.frames[0], "input", "MY_CLASS.init(input:)"
        )
        check_not_captured_error(
            self, thread.frames[0], "find_me", "MY_CLASS.init(input:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

        lldbutil.continue_to_source_breakpoint(
            self, process, "break_static_member_class", lldb.SBFileSpec("main.swift")
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "input_static",
            "static MY_CLASS.static_func(input_static:)",
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "find_me_static",
            "static MY_CLASS.static_func(input_static:)",
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me_static")

        for kind in ["getter", "setter"]:
            lldbutil.continue_to_source_breakpoint(
                self,
                process,
                f"break_class_computed_property_{kind}",
                lldb.SBFileSpec("main.swift"),
            )
            check_not_captured_error(
                self,
                thread.frames[0],
                "find_me",
                f"MY_CLASS.class_computed_property.{kind}",
            )
            check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")
        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            f"break_class_computed_property_didset",
            lldb.SBFileSpec("main.swift"),
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "find_me",
            f"MY_CLASS.class_computed_property_didset.didset",
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    def test_ctor_struct_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_ctor_struct")
        check_not_captured_error(
            self, thread.frames[0], "input", "MY_STRUCT.init(input:)"
        )
        check_not_captured_error(
            self, thread.frames[0], "find_me", "MY_STRUCT.init(input:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

        lldbutil.continue_to_source_breakpoint(
            self, process, "break_static_member_struct", lldb.SBFileSpec("main.swift")
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "input_static",
            "static MY_STRUCT.static_func(input_static:)",
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "find_me_static",
            "static MY_STRUCT.static_func(input_static:)",
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me_static")

        for kind in ["getter", "setter"]:
            lldbutil.continue_to_source_breakpoint(
                self,
                process,
                f"break_struct_computed_property_{kind}",
                lldb.SBFileSpec("main.swift"),
            )
            check_not_captured_error(
                self,
                thread.frames[0],
                "find_me",
                f"MY_STRUCT.struct_computed_property.{kind}",
            )
            check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")
        lldbutil.continue_to_source_breakpoint(
            self,
            process,
            f"break_struct_computed_property_didset",
            lldb.SBFileSpec("main.swift"),
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "find_me",
            f"MY_STRUCT.struct_computed_property_didset.didset",
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    def test_ctor_enum_closure(self):
        self.build()
        (target, process, thread) = self.get_to_bkpt("break_ctor_enum")
        check_not_captured_error(
            self, thread.frames[0], "input", "MY_ENUM.init(input:)"
        )
        check_not_captured_error(
            self, thread.frames[0], "find_me", "MY_ENUM.init(input:)"
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

        lldbutil.continue_to_source_breakpoint(
            self, process, "break_static_member_enum", lldb.SBFileSpec("main.swift")
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "input_static",
            "static MY_ENUM.static_func(input_static:)",
        )
        check_not_captured_error(
            self,
            thread.frames[0],
            "find_me_static",
            "static MY_ENUM.static_func(input_static:)",
        )
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me_static")
