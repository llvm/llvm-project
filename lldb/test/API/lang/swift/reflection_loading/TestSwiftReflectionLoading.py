import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import re

class TestSwiftReflectionLoading(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test that reflection metadata is imported"""
        self.build()

        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'),
            extra_images=['dynamic_lib'])
        frame = thread.frames[0]
        var_c = frame.FindVariable("c")
        var_c_x = var_c.GetChildMemberWithName("x")
        lldbutil.check_variable(self, var_c_x, value="23")

        # Scan through the types log.
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found_exe = 0
        found_lib = 0
        for line in logfile:
            if re.search(r'Adding reflection metadata in .*a\.out', line):
                found_exe += 1
            if re.search(r'Adding reflection metadata in .*dynamic_lib', line):
                found_lib += 1
        self.assertEqual(found_exe, 1)
        self.assertEqual(found_lib, 1)
