import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftNoRuntime(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test running a Swift expression in a C program"""
        self.build()
        self.expect("b test")
        _, process, _, _ = lldbutil.run_to_name_breakpoint(self, "main")
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(False);
        value = self.frame().EvaluateExpression("test()", options)
        self.assertIn("breakpoint", str(value.GetError()))
        process.Kill()

