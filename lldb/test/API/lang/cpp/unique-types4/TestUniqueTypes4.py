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
            "expression ns::Foo<double>::value", substrs=["no member named"], error=True
        )
        self.expect(
            "expression ns::Foo<int>::value", substrs=["no member named"], error=True
        )
        self.expect(
            "expression ns::Bar<double>::value", substrs=["no member named"], error=True
        )
        self.expect(
            "expression ns::Bar<int>::value", substrs=["no member named"], error=True
        )
        self.expect(
            "expression ns::FooDouble::value",
            substrs=["Couldn't lookup symbols"],
            error=True,
        )
        self.expect(
            "expression ns::FooInt::value",
            substrs=["Couldn't lookup symbols"],
            error=True,
        )

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_no_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gno-simple-template-names"))
