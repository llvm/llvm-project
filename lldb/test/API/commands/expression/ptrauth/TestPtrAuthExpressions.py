import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrAuthExpressions(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArm64eSupported
    def test_static_function_pointer(self):
        """On arm64e, function pointers are automatically signed (PAC).
        Test that we can call a function through a static function pointer
        from the expression evaluator, which requires "fixing up" the pointer
        signing via the InjectPointerSigningFixups pass."""
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
    def test_indirect_call_through_caller(self):
        """Test that a function pointer passed to a debuggee function is
        correctly signed. The caller() function in the debuggee forces a
        genuine indirect call, preventing the compiler from folding the
        function pointer call into a direct call."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "caller(add, 2, 3);",
            result_type="int",
            result_value="5",
        )

        self.expect_expr(
            "caller(mul, 3, 7);",
            result_type="int",
            result_value="21",
        )

    @skipUnlessArm64eSupported
    def test_debuggee_signed_pointer(self):
        """Test that a signed function pointer stored in the debuggee's memory
        can be read and called from a user expression. The global_fp variable
        is signed with the IB key (__ptrauth(1, 0, 0)), which is
        process-specific; this verifies that auth succeeds because expressions
        execute in the debuggee's process, not the debugger's."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "global_fp(10, 20);",
            result_type="int",
            result_value="30",
        )
