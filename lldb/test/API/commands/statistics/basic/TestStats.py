import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        self.expect("statistics disable", substrs=['need to enable statistics before disabling'], error=True)

        # 'expression' should change the statistics.
        self.expect("statistics enable")
        self.expect("statistics enable", substrs=['already enabled'], error=True)
        self.expect("expr patatino", substrs=['27'])
        self.expect("statistics disable")
        self.expect("statistics dump", substrs=['expr evaluation successes : 1\n',
                                                'expr evaluation failures : 0\n'])

        self.expect("statistics enable")
        # Doesn't parse.
        self.expect("expr doesnt_exist", error=True,
                    substrs=["undeclared identifier 'doesnt_exist'"])
        # Doesn't successfully execute.
        self.expect("expr int *i = nullptr; *i", error=True)
        # Interpret an integer as an array with 3 elements is also a failure.
        self.expect("expr -Z 3 -- 1", error=True,
                    substrs=["expression cannot be used with --element-count"])
        self.expect("statistics disable")
        # We should have gotten 3 new failures and the previous success.
        self.expect("statistics dump", substrs=['expr evaluation successes : 1\n',
                                                'expr evaluation failures : 3\n'])

        # 'frame var' with disabled statistics shouldn't change stats.
        self.expect("frame var", substrs=['27'])

        self.expect("statistics enable")
        # 'frame var' with enabled statistics will change stats.
        self.expect("frame var", substrs=['27'])
        self.expect("statistics disable")
        self.expect("statistics dump", substrs=['frame var successes : 1\n',
                                                'frame var failures : 0\n'])
