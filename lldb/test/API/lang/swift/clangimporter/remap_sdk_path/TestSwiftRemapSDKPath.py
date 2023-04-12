import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftRewriteClangPaths(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @swiftTest
    def test(self):
        """Test that clang flags pointing into an SDK on a different machine are remapped"""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')
        self.expect("expression 1", substrs=["1"])

        # Scan through the types log.
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found = 0
        for line in logfile:
            if line.startswith(' SwiftASTContextForModule("a.out")::RemapClangImporterOptions() -- remapped'):
                if '/LocalSDK/' in line:
                    found += 1
        self.assertEqual(found, 1)
