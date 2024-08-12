import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftClangImporterExplicitCC1(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(setting=('symbols.swift-precise-compiler-invocation', 'false'))
    @skipIf(setting=('plugin.typesystem.clang.experimental-redecl-completion', 'true'), bugnumber='rdar://128094135')
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """
        Test flipping on/off implicit modules.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("expression obj", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=["b ="])
        self.filecheck('platform shell cat "%s"' % log, __file__)
#       CHECK:  SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration() --     -cc1
