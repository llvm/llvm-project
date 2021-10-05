import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftSystemFramework(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_system_framework(self):
        """Make sure no framework paths into /System/Library are added"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        self.expect("settings set target.use-all-compiler-flags true")
        self.expect("expression -- 0")
        pos = 0
        neg = 0
        with open(log, "r") as logfile:
            for line in logfile:
                if "-- rejecting framework path " in line:
                    pos += 1
                elif ("reflection metadata" not in line) and \
                     ("/System/Library/Frameworks" in line):
                    neg += 1

        self.assertGreater(pos, 0, "sanity check failed")
        self.assertEqual(neg, 0, "found /System/Library/Frameworks in log")
