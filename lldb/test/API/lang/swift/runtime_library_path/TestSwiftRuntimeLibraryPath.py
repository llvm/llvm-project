import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftRuntimeLibraryPath(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)
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

        self.expect("expression 1")
        self.filecheck('platform shell cat "%s"' % log, __file__)
        # CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration(){{.*}}Runtime library paths{{.*}}2 items
