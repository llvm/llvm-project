import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftRuntimeLibraryPath(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    def test(self):
        """Test that the default runtime library path can be recovered even if
        paths weren't serialized."""
        self.build()
        log = self.getBuildArtifact("types.log")
        command_result = lldb.SBCommandReturnObject()
        interpreter = self.dbg.GetCommandInterpreter()
        interpreter.HandleCommand("log enable lldb types -f "+log, command_result)

        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')

        self.expect("p 1")
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        in_expr_log = 0
        found = 0
        for line in logfile:
            if line.startswith(" SwiftASTContextForExpressions::LogConfiguration(SwiftASTContext"):
                in_expr_log += 1
            if in_expr_log and "Runtime library paths" in line and \
               "2 items" in line:
                found += 1
        self.assertEqual(in_expr_log, 1)
        self.assertEqual(found, 1)
