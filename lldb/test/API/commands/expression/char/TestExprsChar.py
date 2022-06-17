import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprCharTestCase(TestBase):

    def do_test(self, dictionary=None):
        """These basic expression commands should work as expected."""
        self.build(dictionary=dictionary)

        lldbutil.run_to_source_breakpoint(self, '// Break here', lldb.SBFileSpec("main.cpp"))

        self.expect_expr("foo(c)", result_value="1")
        self.expect_expr("foo(sc)", result_value="2")
        self.expect_expr("foo(uc)", result_value="3")

    def test_default_char(self):
        self.do_test()

    @skipIf(oslist=["linux"], archs=["aarch64", "arm"], bugnumber="llvm.org/pr23069")
    @expectedFailureAll(
        archs=[
            "powerpc64le",
            "s390x"],
        bugnumber="llvm.org/pr23069")
    def test_signed_char(self):
        self.do_test(dictionary={'CFLAGS_EXTRAS': '-fsigned-char'})

    @expectedFailureAll(
        archs=[
            "i[3-6]86",
            "x86_64",
            "arm64",
            'arm64e',
            'armv7',
            'armv7k',
            'arm64_32'],
        bugnumber="llvm.org/pr23069, <rdar://problem/28721938>")
    @expectedFailureAll(triple='mips*', bugnumber="llvm.org/pr23069")
    def test_unsigned_char(self):
        self.do_test(dictionary={'CFLAGS_EXTRAS': '-funsigned-char'})
