"""
Tests that the expression evaluator traps on ptrauth authentication failures
when -fptrauth-auth-traps is enabled.  Auth traps cause aut* instructions to
be followed by a brk trap that fires on authentication failure.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import configuration


class TestPtrAuthAuthTraps(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def build_arm64e(self):
        self.build(
            dictionary={"TRIPLE": configuration.triple.replace("arm64", "arm64e")}
        )

    @skipUnlessArm64eSupported
    def test_static_function_pointer(self):
        self.build_arm64e()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect(
            "expression -- "
            "static int (*bad)(int, int) = "
            "(int (*)(int, int))__builtin_ptrauth_sign_unauthenticated("
            "__builtin_ptrauth_strip((void *)&add, 0), 0, 42); "
            "bad(5, 6)",
            error=True,
            substrs=["execution was interrupted"],
        )

    @skipUnlessArm64eSupported
    def test_indirect_call_through_caller(self):
        self.build_arm64e()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect(
            "expression -- "
            "int (*bad)(int, int) = "
            "(int (*)(int, int))__builtin_ptrauth_sign_unauthenticated("
            "__builtin_ptrauth_strip((void *)&add, 0), 0, 42); "
            "caller(bad, 2, 3)",
            error=True,
            substrs=["execution was interrupted"],
        )
