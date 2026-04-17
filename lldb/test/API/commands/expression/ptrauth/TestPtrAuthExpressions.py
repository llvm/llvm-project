import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import configuration


class TestPtrAuthExpressions(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def build_arm64e(self):
        self.build(
            dictionary={"TRIPLE": configuration.triple.replace("arm64", "arm64e")}
        )

    @skipUnlessArm64eSupported
    def test_static_function_pointer(self):
        """On arm64e, function pointers are automatically signed (PAC).
        Test that we can call a function through a static function pointer
        from the expression evaluator, which requires "fixing up" the pointer
        signing via the InjectPointerSigningFixups pass."""
        self.build_arm64e()

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
        self.build_arm64e()

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
        self.build_arm64e()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        self.expect_expr(
            "global_fp(10, 20);",
            result_type="int",
            result_value="30",
        )

    @skipUnlessArm64eSupported
    def test_indirect_goto(self):
        """Test that computed gotos (GCC labels-as-values) work in the
        expression evaluator on arm64e, where -fptrauth-indirect-gotos signs
        label addresses and the indirect branch authenticates them."""
        self.build_arm64e()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )

        # Call a debuggee function that uses a computed-goto dispatch table.
        self.expect_expr(
            "indirect_goto_dispatch(0)",
            result_type="int",
            result_value="10",
        )
        self.expect_expr(
            "indirect_goto_dispatch(1)",
            result_type="int",
            result_value="20",
        )
        self.expect_expr(
            "indirect_goto_dispatch(2)",
            result_type="int",
            result_value="30",
        )

        # Evaluate a computed goto directly in a user expression.
        # Use individual variables (not an array) so that the label addresses
        # are signed inline with pacia/braa instructions, avoiding @AUTH
        # relocations in global constant tables that RuntimeDyld cannot handle.
        self.expect_expr(
            "({ int result; void *t0 = &&L0, *t1 = &&L1, *t2 = &&L2; "
            "goto *t1; "
            "L0: result = 100; goto Lend; "
            "L1: result = 200; goto Lend; "
            "L2: result = 300; goto Lend; "
            "Lend: result; })",
            result_type="int",
            result_value="200",
        )
