import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftExpressionErrorReportingy(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test error reporting in expressions reports
        only diagnostics in user code"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        options = lldb.SBExpressionOptions()
        value = self.frame().EvaluateExpression(
            "ceciNestPasUnVar", options)
        def check(value):
            lines = str(value.GetError()).split('\n')
            self.assertTrue(lines[0].startswith('error:'))
            self.assertIn('ceciNestPasUnVar', lines[0])
            for line in lines[1:]:
                self.assertFalse(line.startswith('error:'))
                self.assertFalse(line.startswith('warning:'))

        check(value)
        process.Continue()
        value = self.frame().EvaluateExpression(
            "ceciNestPasUnVar", options)
        check(value)
