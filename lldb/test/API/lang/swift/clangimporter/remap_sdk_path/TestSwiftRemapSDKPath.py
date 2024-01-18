import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftRewriteClangPaths(TestBase):
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
        self.filecheck('platform shell cat "%s"' % log, __file__)
#       CHECK:  SwiftASTContextForExpressions::RemapClangImporterOptions() -- remapped{{.*}}/LocalSDK/
