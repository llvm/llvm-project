import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("f0", result_type="Foo<__bf16>")
        self.expect_expr("f1", result_type="Foo<__fp16>")

        # Test sizeof to ensure while computing layout we don't do
        # infinite recursion.
        v = self.frame().EvaluateExpression("sizeof(f0)")
        self.assertEqual(v.GetValueAsUnsigned() > 0, True)
        v = self.frame().EvaluateExpression("sizeof(f1)")
        self.assertEqual(v.GetValueAsUnsigned() > 0, True)
