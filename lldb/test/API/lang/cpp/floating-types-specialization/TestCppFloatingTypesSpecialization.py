import lldb
import lldbsuite.test.lldbplatformutil as lldbplatformutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipIf(compiler="clang", compiler_version=["<", "17.0"])
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # On 32-bit Arm, you have to have the bfloat16 extension, or an FPU while
        # not using the soft float mode. The target we assume has none of that
        # so instead of __bf16 we get __fp16.
        is_arm_32_bit = lldbplatformutil.getArchitecture() == "arm"

        self.expect_expr(
            "f0", result_type=("Foo<__fp16>" if is_arm_32_bit else "Foo<__bf16>")
        )

        # When __bf16 is actually __fp16, f1 looks like it inherits from itself.
        # Which clang allows but LLDB fails to evaluate.
        if not is_arm_32_bit:
            self.expect_expr("f1", result_type="Foo<__fp16>")

        # Test sizeof to ensure while computing layout we don't do
        # infinite recursion.
        v = self.frame().EvaluateExpression("sizeof(f0)")
        self.assertEqual(v.GetValueAsUnsigned() > 0, True)

        if not is_arm_32_bit:
            v = self.frame().EvaluateExpression("sizeof(f1)")
            self.assertEqual(v.GetValueAsUnsigned() > 0, True)
