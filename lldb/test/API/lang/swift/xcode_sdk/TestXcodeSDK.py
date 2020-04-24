import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAppleInternalSDK(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test(self):
        """Test that we can detect an Xcode SDK from the DW_AT_APPLE_sdk attribute."""
        self.build()
        log = self.getBuildArtifact("types.log")
        command_result = lldb.SBCommandReturnObject()
        interpreter = self.dbg.GetCommandInterpreter()
        interpreter.HandleCommand("log enable lldb types -f "+log, command_result)

        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')

        self.expect("p 1")
        logfile = open(log, "r")
        in_expr_log = 0
        found = 0
        for line in logfile:
            if line.startswith(" SwiftASTContextForExpressions::LogConfiguration"):
                in_expr_log += 1
            if in_expr_log and "SDK path" in line and ".sdk" in line:
                found += 1
        self.assertEqual(in_expr_log, 1)
        self.assertEqual(found, 1)
