import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test_event.build_exception import BuildError


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def build_and_run(self, test_file):
        """
        Tries building the given test source and runs to the first breakpoint.
        Returns false if the file fails to build due to an unsupported calling
        convention on the current test target. Returns true if building and
        running to the breakpoint succeeded.
        """
        try:
            self.build(
                dictionary={
                    "C_SOURCES": test_file,
                    "CFLAGS_EXTRAS": "-Werror=ignored-attributes",
                }
            )
        except BuildError as e:
            # Test source failed to build. Check if it failed because the
            # calling convention argument is unsupported/unknown in which case
            # the test should be skipped.
            error_msg = str(e)
            # Clang gives an explicit error when a calling convention is
            # not supported.
            if "calling convention is not supported for this target" in error_msg:
                return False
            # GCC's has two different generic warnings it can emit.
            if "attribute ignored" in error_msg:
                return False
            if "attribute directive ignored " in error_msg:
                return False
            # We got a different build error, so raise it again to fail the
            # test.
            raise
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec(test_file)
        )
        return True

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    def test_regcall(self):
        if not self.build_and_run("regcall.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    def test_ms_abi(self):
        if not self.build_and_run("ms_abi.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    def test_stdcall(self):
        if not self.build_and_run("stdcall.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")

    @expectedFailureAll(
        oslist=["freebsd"], bugnumber="github.com/llvm/llvm-project/issues/56084"
    )
    def test_vectorcall(self):
        if not self.build_and_run("vectorcall.c"):
            return
        self.expect_expr("func(1.0)", result_type="int", result_value="1")

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    def test_fastcall(self):
        if not self.build_and_run("fastcall.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")

    @skipIf(compiler="clang", compiler_version=["<", "9.0"])
    def test_pascal(self):
        if not self.build_and_run("pascal.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")

    def test_sysv_abi(self):
        if not self.build_and_run("sysv_abi.c"):
            return
        self.expect_expr("func(1, 2, 3, 4)", result_type="int", result_value="10")
