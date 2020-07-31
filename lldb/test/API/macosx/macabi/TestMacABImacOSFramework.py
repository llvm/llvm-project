import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestMacABImacOSFramework(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(macos_version=["<", "10.15"])
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    # There is a Clang driver change missing on llvm.org.
    @expectedFailureAll(bugnumber="rdar://problem/54986190>")
    @skipIfReproducer # This is hitting https://bugs.python.org/issue22393
    def test_macabi(self):
        """Test the x86_64-apple-ios-macabi target linked against a macos dylib"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.c'))
        self.expect("image list -t -b",
                    patterns=["x86_64.*-apple-ios.*-macabi a\.out",
                              "x86_64.*-apple-macosx.* libfoo.dylib[^(]"])
        self.expect("fr v s", "Hello MacABI")
        self.expect("p s", "Hello MacABI")
