"""
Test the printing of anonymous and named namespace variables.
"""


import unittest
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NamespaceLookupTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Break inside different scopes and evaluate value
        self.line_break_global_scope = line_number("ns.cpp", "// BP_global_scope")
        self.line_break_file_scope = line_number("ns2.cpp", "// BP_file_scope")
        self.line_break_ns_scope = line_number("ns2.cpp", "// BP_ns_scope")
        self.line_break_nested_ns_scope = line_number(
            "ns2.cpp", "// BP_nested_ns_scope"
        )
        self.line_break_nested_ns_scope_after_using = line_number(
            "ns2.cpp", "// BP_nested_ns_scope_after_using"
        )
        self.line_break_before_using_directive = line_number(
            "ns3.cpp", "// BP_before_using_directive"
        )
        self.line_break_after_using_directive = line_number(
            "ns3.cpp", "// BP_after_using_directive"
        )

    def runToBkpt(self, command):
        self.runCmd(command, RUN_SUCCEEDED)
        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    @skipIfWindows  # This is flakey on Windows: llvm.org/pr38373
    @unittest.expectedFailure  # CU-local objects incorrectly scoped
    def test_scope_lookup_with_run_command_globals(self):
        """Test scope lookup of functions in lldb."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, self.line_break_global_scope, lldb.SBFileSpec("ns.cpp")
        )

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_before_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_global_scope at file scope
        self.runToBkpt("run")

        # FIXME: LLDB does not correctly scope CU-local objects.
        # LLDB currently lumps functions from all files into
        # a single AST and depending on the order with which
        # functions are considered, LLDB can incorrectly call
        # the static local ns.cpp::func() instead of the expected
        # ::func()

        # Evaluate func() - should call ::func()
        self.expect_expr("func()", expect_type="int", expect_value="1")

        # Evaluate ::func() - should call A::func()
        self.expect_expr("::func()", result_type="int", result_value="1")

        # Continue to BP_before_using_directive at file scope
        self.runToBkpt("continue")

        # Evaluate func() - should call ::func()
        self.expect_expr("func()", result_type="int", result_value="1")

        # Evaluate ::func() - should call ::func()
        self.expect_expr("::func()", result_type="int", result_value="1")

        # Continue to BP_after_using_directive at file scope
        self.runToBkpt("continue")

        # Evaluate ::func() - should call ::func()
        self.expect_expr("::func()", result_type="int", result_value="1")

    @skipIfWindows  # This is flakey on Windows: llvm.org/pr38373
    def test_scope_lookup_with_run_command(self):
        """Test scope lookup of functions in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns.cpp",
            self.line_break_global_scope,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_ns_scope,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope_after_using,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_file_scope,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_before_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_global_scope at global scope
        self.runToBkpt("run")

        # Evaluate A::B::func() - should call A::B::func()
        self.expect_expr("A::B::func()", result_type="int", result_value="4")
        # Evaluate func(10) - should call ::func(int)
        self.expect_expr("func(10)", result_type="int", result_value="11")
        # Evaluate A::foo() - should call A::foo()
        self.expect_expr("A::foo()", result_type="int", result_value="42")

        # Continue to BP_file_scope at file scope
        self.runToBkpt("continue")
        # FIXME: In DWARF 5 with dsyms, the ordering of functions is slightly
        # different, which also hits the same issues mentioned previously.
        if configuration.dwarf_version <= 4 or self.getDebugInfo() == "dwarf":
            self.expect_expr("func()", result_type="int", result_value="2")

        # Continue to BP_ns_scope at ns scope
        self.runToBkpt("continue")
        # Evaluate func(10) - should call A::func(int)
        self.expect_expr("func(10)", result_type="int", result_value="13")
        # Evaluate B::func() - should call B::func()
        self.expect_expr("B::func()", result_type="int", result_value="4")
        # Evaluate func() - should call A::func()
        self.expect_expr("func()", result_type="int", result_value="3")

        # Continue to BP_nested_ns_scope at nested ns scope
        self.runToBkpt("continue")
        # Evaluate func() - should call A::B::func()
        self.expect_expr("func()", result_type="int", result_value="4")
        # Evaluate A::func() - should call A::func()
        self.expect_expr("A::func()", result_type="int", result_value="3")

        # Evaluate func(10) - should call A::func(10)
        # NOTE: Under the rules of C++, this test would normally get an error
        # because A::B::func() hides A::func(), but lldb intentionally
        # disobeys these rules so that the intended overload can be found
        # by only removing duplicates if they have the same type.
        self.expect_expr("func(10)", result_type="int", result_value="13")

        # Continue to BP_nested_ns_scope_after_using at nested ns scope after
        # using declaration
        self.runToBkpt("continue")
        # Evaluate A::func(10) - should call A::func(int)
        self.expect_expr("A::func(10)", result_type="int", result_value="13")

        # Continue to BP_before_using_directive at global scope before using
        # declaration
        self.runToBkpt("continue")
        # Evaluate B::func() - should call B::func()
        self.expect_expr("B::func()", result_type="int", result_value="4")

        # Continue to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("continue")
        # Evaluate B::func() - should call B::func()
        self.expect_expr("B::func()", result_type="int", result_value="4")

    @unittest.expectedFailure  # lldb scope lookup of functions bugs
    def test_function_scope_lookup_with_run_command(self):
        """Test scope lookup of functions in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns.cpp",
            self.line_break_global_scope,
            num_expected_locations=1,
            loc_exact=False,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_ns_scope,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_global_scope at global scope
        self.runToBkpt("run")
        # Evaluate foo() - should call ::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions.
        self.expect_expr("foo()", result_type="int", result_value="42")
        # Evaluate ::foo() - should call ::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions and :: is ignored.
        self.expect_expr("::foo()", result_type="int", result_value="42")

        # Continue to BP_ns_scope at ns scope
        self.runToBkpt("continue")
        # Evaluate foo() - should call A::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions.
        self.expect_expr("foo()", result_type="int", result_value="42")

    # NOTE: this test may fail on older systems that don't emit import
    # entries in DWARF - may need to add checks for compiler versions here.
    @skipIf(compiler="gcc", oslist=["linux"], debug_info=["dwo"])  # Skip to avoid crash
    def test_scope_after_using_directive_lookup_with_run_command(self):
        """Test scope lookup after using directive in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func2() - should call A::func2()
        self.expect_expr("func2()", result_type="int", result_value="3")

    @unittest.expectedFailure  # lldb scope lookup after using declaration bugs
    # NOTE: this test may fail on older systems that don't emit import
    # emtries in DWARF - may need to add checks for compiler versions here.
    def test_scope_after_using_declaration_lookup_with_run_command(self):
        """Test scope lookup after using declaration in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope_after_using,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_nested_ns_scope_after_using at nested ns scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func() - should call A::func()
        self.expect_expr("func()", result_type="int", result_value="3")

    @unittest.expectedFailure  # lldb scope lookup ambiguity after using bugs
    def test_scope_ambiguity_after_using_lookup_with_run_command(self):
        """Test scope lookup ambiguity after using in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func() - should get error: ambiguous
        # FIXME: This test fails because lldb removes duplicates if they have
        # the same type.
        self.expect("expr -- func()", startstr="error")

    def test_scope_lookup_shadowed_by_using_with_run_command(self):
        """Test scope lookup shadowed by using in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope,
            num_expected_locations=1,
            loc_exact=False,
        )

        # Run to BP_nested_ns_scope at nested ns scope
        self.runToBkpt("run")
        # Evaluate func(10) - should call A::func(10)
        # NOTE: Under the rules of C++, this test would normally get an error
        # because A::B::func() shadows A::func(), but lldb intentionally
        # disobeys these rules so that the intended overload can be found
        # by only removing duplicates if they have the same type.
        self.expect_expr("func(10)", result_type="int", result_value="13")
