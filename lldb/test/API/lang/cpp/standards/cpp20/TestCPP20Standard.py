import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCPP20Standard(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=["<", "11.0"])
    def test_cpp20(self):
        """
        Tests that we can evaluate an expression in C++20 mode
        """
        self.build()
        lldbutil.run_to_source_breakpoint(self, "Foo{}", lldb.SBFileSpec("main.cpp"))

        self.expect(
            "expr -l c++11 -- Foo{} <=> Foo{}",
            error=True,
            substrs=[
                "'<=>' is a single token in C++20; add a space to avoid a change in behavior"
            ],
        )

        self.expect("expr -l c++20 -- Foo{} <=> Foo{}", substrs=["(bool) $0 = true"])
