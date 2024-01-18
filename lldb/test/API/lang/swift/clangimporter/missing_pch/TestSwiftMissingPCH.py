import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import os

class TestSwiftMissingVFSOverlay(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=("symbols.use-swift-clangimporter", "false"))
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
        # FIXME: This crashes the compiler while trying to diagnose the
        # missing file (because the source location is inside the missing file).
        #self.expect("expr y", substrs=["1"])
