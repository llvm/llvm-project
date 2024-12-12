import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import re

class TestSwiftReflectionLoading(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test(self):
        """Test that reflection metadata is imported"""
        self.build()

        types_log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -v -f "%s"' % types_log)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'),
            extra_images=['dynamic_lib'])
        frame = thread.frames[0]
        var_c = frame.FindVariable("c")
        var_c_x = var_c.GetChildMemberWithName("x")
        lldbutil.check_variable(self, var_c_x, value="23")

        # Scan through the types log.
        self.filecheck('platform shell cat "%s"' % types_log, __file__)
        # CHECK: {{Adding reflection metadata in .*a\.out}}
        # CHECK: {{Adding reflection metadata in .*dynamic_lib}}
