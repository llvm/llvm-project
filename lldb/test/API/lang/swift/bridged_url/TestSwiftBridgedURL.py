import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @requireNotEmbeddedSwift
    @skipUnlessFoundationEssentials
    @skipIfLinux  # https://github.com/swiftlang/llvm-project/issues/13465
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect("v url", patterns=[r'url = "file:///tmp/?"'])
