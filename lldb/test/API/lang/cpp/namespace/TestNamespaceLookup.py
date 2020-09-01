"""
Test the printing of anonymous and named namespace variables.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NamespaceLookupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Break inside different scopes and evaluate value
        self.line_break_global_scope = line_number(
            'ns.cpp', '// BP_global_scope')
        self.line_break_file_scope = line_number('ns2.cpp', '// BP_file_scope')
        self.line_break_ns_scope = line_number('ns2.cpp', '// BP_ns_scope')
        self.line_break_nested_ns_scope = line_number(
            'ns2.cpp', '// BP_nested_ns_scope')
        self.line_break_nested_ns_scope_after_using = line_number(
            'ns2.cpp', '// BP_nested_ns_scope_after_using')
        self.line_break_before_using_directive = line_number(
            'ns3.cpp', '// BP_before_using_directive')
        self.line_break_after_using_directive = line_number(
            'ns3.cpp', '// BP_after_using_directive')

    def runToBkpt(self, command):
        self.runCmd(command, RUN_SUCCEEDED)
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr25819")
    @skipIfWindows # This is flakey on Windows: llvm.org/pr38373
    def test_scope_lookup_with_run_command(self):
        """Test scope lookup of functions in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns.cpp",
            self.line_break_global_scope,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_ns_scope,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope_after_using,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_before_using_directive,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_global_scope at global scope
        self.runToBkpt("run")
        # Evaluate func() - should call ::func()
        self.expect("expr -- func()", startstr="(int) $0 = 1")
        # Evaluate A::B::func() - should call A::B::func()
        self.expect("expr -- A::B::func()", startstr="(int) $1 = 4")
        # Evaluate func(10) - should call ::func(int)
        self.expect("expr -- func(10)", startstr="(int) $2 = 11")
        # Evaluate ::func() - should call A::func()
        self.expect("expr -- ::func()", startstr="(int) $3 = 1")
        # Evaluate A::foo() - should call A::foo()
        self.expect("expr -- A::foo()", startstr="(int) $4 = 42")

        # Continue to BP_ns_scope at ns scope
        self.runToBkpt("continue")
        # Evaluate func(10) - should call A::func(int)
        self.expect("expr -- func(10)", startstr="(int) $5 = 13")
        # Evaluate B::func() - should call B::func()
        self.expect("expr -- B::func()", startstr="(int) $6 = 4")
        # Evaluate func() - should call A::func()
        self.expect("expr -- func()", startstr="(int) $7 = 3")

        # Continue to BP_nested_ns_scope at nested ns scope
        self.runToBkpt("continue")
        # Evaluate func() - should call A::B::func()
        self.expect("expr -- func()", startstr="(int) $8 = 4")
        # Evaluate A::func() - should call A::func()
        self.expect("expr -- A::func()", startstr="(int) $9 = 3")

        # Evaluate func(10) - should call A::func(10)
        # NOTE: Under the rules of C++, this test would normally get an error
        # because A::B::func() hides A::func(), but lldb intentionally
        # disobeys these rules so that the intended overload can be found
        # by only removing duplicates if they have the same type.
        self.expect("expr -- func(10)", startstr="(int) $10 = 13")

        # Continue to BP_nested_ns_scope_after_using at nested ns scope after
        # using declaration
        self.runToBkpt("continue")
        # Evaluate A::func(10) - should call A::func(int)
        self.expect("expr -- A::func(10)", startstr="(int) $11 = 13")

        # Continue to BP_before_using_directive at global scope before using
        # declaration
        self.runToBkpt("continue")
        # Evaluate ::func() - should call ::func()
        self.expect("expr -- ::func()", startstr="(int) $12 = 1")
        # Evaluate B::func() - should call B::func()
        self.expect("expr -- B::func()", startstr="(int) $13 = 4")

        # Continue to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("continue")
        # Evaluate ::func() - should call ::func()
        self.expect("expr -- ::func()", startstr="(int) $14 = 1")
        # Evaluate B::func() - should call B::func()
        self.expect("expr -- B::func()", startstr="(int) $15 = 4")

    @expectedFailure("lldb scope lookup of functions bugs")
    def test_function_scope_lookup_with_run_command(self):
        """Test scope lookup of functions in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns.cpp",
            self.line_break_global_scope,
            num_expected_locations=1,
            loc_exact=False)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_ns_scope,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_global_scope at global scope
        self.runToBkpt("run")
        # Evaluate foo() - should call ::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions.
        self.expect("expr -- foo()", startstr="(int) $0 = 42")
        # Evaluate ::foo() - should call ::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions and :: is ignored.
        self.expect("expr -- ::foo()", startstr="(int) $1 = 42")

        # Continue to BP_ns_scope at ns scope
        self.runToBkpt("continue")
        # Evaluate foo() - should call A::foo()
        # FIXME: lldb finds Y::foo because lookup for variables is done
        # before functions.
        self.expect("expr -- foo()", startstr="(int) $2 = 42")

    @expectedFailure("lldb file scope lookup bugs")
    @skipIfWindows # This is flakey on Windows: llvm.org/pr38373
    def test_file_scope_lookup_with_run_command(self):
        """Test file scope lookup in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_file_scope,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_file_scope at file scope
        self.runToBkpt("run")
        # Evaluate func() - should call static ns2.cpp:func()
        # FIXME: This test fails because lldb doesn't know about file scopes so
        # finds the global ::func().
        self.expect("expr -- func()", startstr="(int) $0 = 2")

    @skipIfWindows # This is flakey on Windows: llvm.org/pr38373
    def test_scope_lookup_before_using_with_run_command(self):
        """Test scope lookup before using in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_before_using_directive,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_before_using_directive at global scope before using
        # declaration
        self.runToBkpt("run")
        # Evaluate func() - should call ::func()
        self.expect("expr -- func()", startstr="(int) $0 = 1")

    # NOTE: this test may fail on older systems that don't emit import
    # entries in DWARF - may need to add checks for compiler versions here.
    @skipIf(
        compiler="gcc",
        oslist=["linux"],
        debug_info=["dwo"])  # Skip to avoid crash
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr25819")
    def test_scope_after_using_directive_lookup_with_run_command(self):
        """Test scope lookup after using directive in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func2() - should call A::func2()
        self.expect("expr -- func2()", startstr="(int) $0 = 3")

    @expectedFailure(
        "lldb scope lookup after using declaration bugs")
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
            loc_exact=False)

        # Run to BP_nested_ns_scope_after_using at nested ns scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func() - should call A::func()
        self.expect("expr -- func()", startstr="(int) $0 = 3")

    @expectedFailure("lldb scope lookup ambiguity after using bugs")
    def test_scope_ambiguity_after_using_lookup_with_run_command(self):
        """Test scope lookup ambiguity after using in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns3.cpp",
            self.line_break_after_using_directive,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_after_using_directive at global scope after using
        # declaration
        self.runToBkpt("run")
        # Evaluate func() - should get error: ambiguous
        # FIXME: This test fails because lldb removes duplicates if they have
        # the same type.
        self.expect("expr -- func()", startstr="error")

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr25819")
    def test_scope_lookup_shadowed_by_using_with_run_command(self):
        """Test scope lookup shadowed by using in lldb."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "ns2.cpp",
            self.line_break_nested_ns_scope,
            num_expected_locations=1,
            loc_exact=False)

        # Run to BP_nested_ns_scope at nested ns scope
        self.runToBkpt("run")
        # Evaluate func(10) - should call A::func(10)
        # NOTE: Under the rules of C++, this test would normally get an error
        # because A::B::func() shadows A::func(), but lldb intentionally
        # disobeys these rules so that the intended overload can be found
        # by only removing duplicates if they have the same type.
        self.expect("expr -- func(10)", startstr="(int) $0 = 13")
