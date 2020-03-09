"""
Test basic std::list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicList(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (size_t)a.size()", substrs=['(size_t) $0 = 3'])
        self.expect("expr (int)a.front()", substrs=['(int) $1 = 3'])
        self.expect("expr (int)a.back()", substrs=['(int) $2 = 2'])

        self.expect("expr a.sort()")
        self.expect("expr (int)a.front()", substrs=['(int) $3 = 1'])
        self.expect("expr (int)a.back()", substrs=['(int) $4 = 3'])

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect("expr (int)a.front()", substrs=['(int) $5 = 3'])
        self.expect("expr (int)a.back()", substrs=['(int) $6 = 1'])

        self.expect("expr (int)(*a.begin())", substrs=['(int) $7 = 3'])
        self.expect("expr (int)(*a.rbegin())", substrs=['(int) $8 = 1'])

