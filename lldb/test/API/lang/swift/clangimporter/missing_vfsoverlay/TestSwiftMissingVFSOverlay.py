import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftMissingVFSOverlay(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """This used to be a test for a diagnostic, however,
        this is no longer an unrecoverable error"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=["Foo"]
        )
        self.expect("expr y", substrs=["42"])
