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
        return lldbutil.run_to_source_breakpoint(
            self, bkpt_name, lldb.SBFileSpec("main.swift")
        )

    @swiftTest
    def test_simple_closure(self):
        self.build()
        (target, process, thread, bkpt) = self.get_to_bkpt("break_simple_closure")
        check_not_captured_error(self, thread.frames[0], "var_in_foo", "func_1(arg:)")
        check_not_captured_error(self, thread.frames[0], "arg", "func_1(arg:)")
        check_no_enhanced_diagnostic(self, thread.frames[0], "dont_find_me")

    @swiftTest
    def test_nested_closure(self):
        self.build()
        (target, process, thread, bkpt) = self.get_to_bkpt("break_double_closure_1")
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
        (target, process, thread, bkpt) = self.get_to_bkpt("break_async_closure_1")
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
