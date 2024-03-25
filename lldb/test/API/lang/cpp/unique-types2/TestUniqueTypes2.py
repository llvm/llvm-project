"""
Test that we return only the requested template instantiation.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class UniqueTypesTestCase2(TestBase):
    def do_test(self, debug_flags):
        """Test that we only display the requested Foo instantiation, not all Foo instantiations."""
        self.build(dictionary=debug_flags)
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "image lookup -A -t '::Foo<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<Foo<int> >'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<float>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )

        self.expect(
            "image lookup -A -t '::FooPack<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<Foo<int> >'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<char, int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<char, float>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<int, int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<int, int, int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'FooPack<float>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )
        self.expect(
            "image lookup -A -t 'FooPack<float, int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )

        self.expect(
            "image lookup -A -t '::Foo<int>::Nested<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<int>::Nested<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t 'Foo<char>::Nested<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )
        self.expect(
            "image lookup -A -t 'Foo<int>::Nested<int>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )
        self.expect(
            "image lookup -A -t 'Nested<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )
        self.expect(
            "image lookup -A -t '::Nested<char>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            error=True,
        )
        self.expect(
            "image lookup -A -t 'Foo<int>::Nested<ns::Bar>'",
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=["1 match found"],
        )

        self.expect_expr("t1", result_type="Foo<char>")
        self.expect_expr("t1", result_type="Foo<char>")
        self.expect_expr("t2", result_type="Foo<int>")
        self.expect_expr("t3", result_type="Foo<Foo<int> >")
        self.expect_expr("p1", result_type="FooPack<char>")
        self.expect_expr("p2", result_type="FooPack<int>")
        self.expect_expr("p3", result_type="FooPack<Foo<int> >")
        self.expect_expr("p4", result_type="FooPack<char, int>")
        self.expect_expr("p5", result_type="FooPack<char, float>")
        self.expect_expr("p6", result_type="FooPack<int, int>")
        self.expect_expr("p7", result_type="FooPack<int, int, int>")
        self.expect_expr("n1", result_type="Foo<int>::Nested<char>")
        self.expect_expr("n2", result_type="Foo<int>::Nested<ns::Bar>")

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gsimple-template-names"))

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_no_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gno-simple-template-names"))
