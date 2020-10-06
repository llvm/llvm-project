"""
Test the output of `frame diagnose` for dereferencing a function argument
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDiagnoseDereferenceArgument(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIf(archs=no_match(['x86_64'])) # <rdar://problem/33842388> frame diagnose doesn't work for armv7 or arm64
    def test_diagnose_dereference_argument(self):
        TestBase.setUp(self)
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("thread list", "Thread should be stopped",
                    substrs=['stopped'])
        self.expect(
            "frame diagnose",
            "Crash diagnosis was accurate",
            substrs=["f->b->d"])
