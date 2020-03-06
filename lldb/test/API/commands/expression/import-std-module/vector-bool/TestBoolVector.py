"""
Test basic std::vector<bool> functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBoolVector(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (size_t)a.size()", substrs=['(size_t) $0 = 4'])
        self.expect("expr (bool)a.front()", substrs=['(bool) $1 = false'])
        self.expect("expr (bool)a[1]", substrs=['(bool) $2 = true'])
        self.expect("expr (bool)a.back()", substrs=['(bool) $3 = true'])

        self.expect("expr (bool)(*a.begin())", substrs=['(bool) $4 = false'])
        self.expect("expr (bool)(*a.rbegin())", substrs=['(bool) $5 = true'])

