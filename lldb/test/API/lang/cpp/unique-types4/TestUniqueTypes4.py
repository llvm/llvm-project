"""
Test that we return only the requested template instantiation.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class UniqueTypesTestCase4(TestBase):
    def do_test(self, debug_flags):
        """Test that we display the correct template instantiation."""
        self.build(dictionary=debug_flags)
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        # FIXME: these should successfully print the values
        self.expect(
            "expression ns::Foo<double>::value",
            substrs=["'Foo' in namespace 'ns'"],
            error=True,
        )
        self.expect(
            "expression ns::Foo<int>::value",
            substrs=["'Foo' in namespace 'ns'"],
            error=True,
        )
        self.expect(
            "expression ns::Bar<double>::value",
            substrs=["'Bar' in namespace 'ns'"],
            error=True,
        )
        self.expect(
            "expression ns::Bar<int>::value",
            substrs=["'Bar' in namespace 'ns'"],
            error=True,
        )
        self.expect_expr("ns::FooDouble::value", result_type="double", result_value="0")
        self.expect_expr("ns::FooInt::value", result_type="int", result_value="0")

    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/75936
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))

    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/75936
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_no_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gno-simple-template-names"))
