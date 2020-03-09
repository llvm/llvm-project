"""
Tests that auto types work
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPAutoTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        compiler="gcc",
        bugnumber="GCC generates incomplete debug info")
    def test_with_run_command(self):
        """Test that auto types work in the expression parser"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr('auto f = 123456; f', result_type='int', result_value='123456')
        self.expect(
            'expr struct Test { int x; int y; Test() : x(123), y(456) {} }; auto t = Test(); t',
            substrs=[
                'Test',
                '123',
                '456'])
        self.expect_expr("auto s = foo; s", result_type="long", result_value="1234")
