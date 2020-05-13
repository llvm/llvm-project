import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftMacroConflict(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True
    
    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test that a broken Clang command line option is diagnosed
           in the expression evaluator"""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.registerSharedLibrariesWithTarget(target, ['Foo'])

        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect("expr y", error=True, substrs=["overlay.yaml", "IRGen"])
