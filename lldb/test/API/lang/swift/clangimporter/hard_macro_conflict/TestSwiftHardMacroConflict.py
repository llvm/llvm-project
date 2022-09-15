import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import shutil

class TestSwiftHardMacroConflict(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    NO_DEBUG_INFO_TESTCASE = True

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['Framework.framework'])
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Framework.swift'))
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        # This is expected to succeed because ClangImporter was set up
        # with the flags from the main executable.
        self.expect("expr bar", "expected result", substrs=["42"])

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)
        self.expect("p foo", error=True)

        per_module_fallback = 0
        import io
        with open(log, "r", encoding='utf-8') as logfile:
            for line in logfile:
                if 'SwiftASTContextForExpressions("Framework")' in line:
                    per_module_fallback += 1
        self.assertGreater(per_module_fallback, 0,
                           "failed to create per-module scratch context")
