import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftSystemFramework(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_system_framework(self):
        """Test the discovery of framework search paths from framework dependencies."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        self.expect("expression -- 0")
        pos = 0
        import io
        with open(log, "r", encoding='utf-8') as logfile:
            for line in logfile:
                if "SwiftASTContextForExpressions::LogConfiguration()" in line and \
                   "/secret_path" in line:
                    pos += 1
        self.assertEqual(pos, 1, "framework search path discovery is broken")
