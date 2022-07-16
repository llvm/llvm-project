import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftReflectionLoading(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test that no SwiftASTContext is initialized just to stop at a breakpoint"""
        self.build()

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        # Scan through the types log.
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found_typeref = 0
        found_astctx = 0
        for line in logfile:
            if 'TypeSystemSwiftTypeRef("a.out")::TypeSystemSwiftTypeRef()' in line:
                found_typeref += 1
            if 'SwiftASTContext' in line:
                found_astctx += 1
        self.assertEqual(found_typeref, 1)
        self.assertEqual(found_astctx, 0)
