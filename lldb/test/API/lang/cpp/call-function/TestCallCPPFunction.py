"""
Tests calling a function by basename
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CallCPPFunctionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number('main.cpp', '// breakpoint')

    def test_with_run_command(self):
        """Test calling a function by basename"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// breakpoint", lldb.SBFileSpec("main.cpp"))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list",
                    STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect_expr("a_function_to_call()", result_type="int", result_value="0")
