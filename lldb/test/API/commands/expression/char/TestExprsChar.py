import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCharTestCase(TestBase):
    def do_test(self, dictionary=None):
        """These basic expression commands should work as expected."""
        self.build(dictionary=dictionary)

        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("foo(c)", result_value="1")
        self.expect_expr("foo(sc)", result_value="2")
        self.expect_expr("foo(uc)", result_value="3")
        self.expect_expr("g", result_type="char")
        self.expect_expr("gs", result_type="signed char")
        self.expect_expr("gu", result_type="unsigned char")

    def test_default_char(self):
        self.do_test()

    def test_signed_char(self):
        self.do_test(dictionary={"CFLAGS_EXTRAS": "-fsigned-char"})

    def test_unsigned_char(self):
        self.do_test(dictionary={"CFLAGS_EXTRAS": "-funsigned-char"})
