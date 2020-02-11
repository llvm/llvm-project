"""
Test std::weak_ptr functionality with a decl from debug info as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestDbgInfoContentWeakPtr(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (int)w.lock()->a", substrs=['(int) $0 = 3'])
        self.expect("expr (int)(w.lock()->a = 5)", substrs=['(int) $1 = 5'])
        self.expect("expr (int)w.lock()->a", substrs=['(int) $2 = 5'])
        self.expect("expr w.use_count()", substrs=['(long) $3 = 1'])
        self.expect("expr w.reset()")
        self.expect("expr w.use_count()", substrs=['(long) $4 = 0'])

