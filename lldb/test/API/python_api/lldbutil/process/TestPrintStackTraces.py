"""
Test SBprocess and SBThread APIs with printing of the stack traces using lldbutil.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadsStackTracesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_stack_traces(self):
        """Test SBprocess and SBThread APIs with printing of the stack traces."""
        self.build()
        (_, process, _, _) = lldbutil.run_to_source_breakpoint(self,
                "// BREAK HERE", lldb.SBFileSpec("main.cpp"))
        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
                    substrs=['(int)x=4', '(int)y=6', '(int)x=3', '(int)argc=1'])
