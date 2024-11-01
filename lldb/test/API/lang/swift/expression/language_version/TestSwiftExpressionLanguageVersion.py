import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExpressionLanguageVersion(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test changing the Swift language version"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')

        expr = """\
#if swift(>=6.0)
6
#else
5
#endif
        """

        def test_version(n):
            if self.TraceOn():
                print("Testing version %d"%n)
            options = lldb.SBExpressionOptions()
            options.SetLanguage(lldb.eLanguageNameSwift, n*100 + 0)
            value = self.frame().EvaluateExpression(expr, options)
            self.assertEqual(value.GetValue(), "%d" % n)

        test_version(5)
        test_version(6)
