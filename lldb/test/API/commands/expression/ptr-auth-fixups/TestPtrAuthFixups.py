import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrAuthFixups(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArm64eSupported
    def test_static_function_pointer(self):
        """On arm64e, function pointers are automatically signed (PAC).
        Test that we can call a function through a function pointer from the
        expression evaluator, which requires "fixing up" the pointer signing."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "static int (*fp)(int, int) = &add; fp(5, 6);",
            result_type="int",
            result_value="11",
        )

        self.expect_expr(
            "static int (*fp)(int, int) = &mul; fp(4, 5);",
            result_type="int",
            result_value="20",
        )

    @skipUnlessArm64eSupported
    def test_local_function_pointer(self):
        """Test that function pointers with automatic (scoped) storage duration
        work correctly. These exercise the PointerAuthCalls codegen path where
        pointers are implicitly signed via codegen, rather than through the
        InjectPointerSigningFixups pass."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "int (*fp)(int, int) = &add; fp(2, 3);",
            result_type="int",
            result_value="5",
        )

        self.expect_expr(
            "int (*fp)(int, int) = &mul; fp(3, 7);",
            result_type="int",
            result_value="21",
        )

    @skipUnlessArm64eSupported
    def test_debuggee_signed_pointer(self):
        """Test that a signed function pointer stored in the debuggee's memory
        can be read and called from a user expression. The global_fp variable
        in the debuggee holds a pointer signed with the debuggee's keys; since
        expressions execute in the debuggee's process, auth should succeed."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "global_fp(10, 20);",
            result_type="int",
            result_value="30",
        )
