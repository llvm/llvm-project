import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestTemplateAlias(TestBase):
    def do_test(self, extra_flags):
        self.build(dictionary=extra_flags)
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("f1", result_type="Foo<int>")
        self.expect_expr("f2", result_type="Foo<double>")
        self.expect_expr("b1", result_type="Bar<int>")
        self.expect_expr("b2", result_type="Bar<double>")
        self.expect_expr("bf1", result_type="Bar<int>")
        self.expect_expr("bf2", result_type="Bar<double>")
        self.expect_expr("bf1", result_type="Bar<int>")
        self.expect_expr("bf2", result_type="Bar<double>")
        self.expect_expr("cbf1", result_type="Container<int>")

    @skipIf(compiler="clang", compiler_version=["<", "21"])
    @expectedFailureAll(
        bugnumber="LLDB doesn't reconstruct template alias names from template parameters"
    )
    def test_tag_alias_simple(self):
        self.do_test(
            dict(CXXFLAGS_EXTRAS="-gdwarf-5 -gtemplate-alias -gsimple-template-names")
        )

    @skipIf(compiler="clang", compiler_version=["<", "21"])
    def test_tag_alias_no_simple(self):
        self.do_test(
            dict(
                CXXFLAGS_EXTRAS="-gdwarf-5 -gtemplate-alias -gno-simple-template-names"
            )
        )

    @skipIf(compiler="clang", compiler_version=["<", "21"])
    def test_no_tag_alias_simple(self):
        self.do_test(
            dict(
                CXXFLAGS_EXTRAS="-gdwarf-5 -gno-template-alias -gsimple-template-names"
            )
        )

    @skipIf(compiler="clang", compiler_version=["<", "21"])
    def test_no_tag_alias_no_simple(self):
        self.do_test(
            dict(
                CXXFLAGS_EXTRAS="-gdwarf-5 -gno-template-alias -gno-simple-template-names"
            )
        )
