import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()

        (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(
            self, "return", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.GetSelectedFrame()

        v1 = self.frame().EvaluateExpression("test")
        v2 = self.frame().EvaluateExpression("i")

        self.assertFalse(v1.IsValid())
        self.assertTrue(v1.GetError().Fail())
        self.assertTrue(v2.IsValid())
