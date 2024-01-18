import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftExtraClangFlags(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    
    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=['windows'])
    @swiftTest
    def test_extra_clang_flags(self):
        """
        Test error handling when ClangImporter emits diagnostics.
        """
        self.build()
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.swift-extra-clang-flags"))
        self.expect('settings set -- target.swift-extra-clang-flags '+
                    '"-DBREAK_STUFF"')
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("expr 0", error=True,
                    substrs=['failed to import bridging header'])
