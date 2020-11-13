import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.m"))

        # Try importing our custom module. This will fail as LLDB won't define
        # the CLANG_ONLY define when it compiles the module for the expression
        # evaluator.
        # Check that the error message shows file/line/column, prints the relevant
        # line from the source code and mentions the module that failed to build.
        self.expect("expr @import LLDBTestModule", error=True,
                    substrs=["module.h:4:1: error: unknown type name 'syntax_error_for_lldb_to_find'",
                             "syntax_error_for_lldb_to_find // comment that tests source printing",
                             "could not build module 'LLDBTestModule'"])
