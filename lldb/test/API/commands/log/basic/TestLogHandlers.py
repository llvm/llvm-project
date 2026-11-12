"""
Test lldb log handlers.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LogHandlerTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.log_file = self.getBuildArtifact("log-file.txt")
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_circular(self):
        self.runCmd("log enable -b 5 -h circular lldb commands")
        self.runCmd("bogus", check=False)
        self.runCmd("log dump lldb -f {}".format(self.log_file))

        with open(self.log_file, "r") as f:
            log_lines = f.readlines()

        self.assertEqual(len(log_lines), 5)

        found_command_log_dump = False
        found_command_bogus = False

        for line in log_lines:
            if "Processing command: log dump" in line:
                found_command_log_dump = True
            if "Processing command: bogus" in line:
                found_command_bogus = True

        self.assertTrue(found_command_log_dump)
        self.assertFalse(found_command_bogus)

    def test_circular_no_buffer_size(self):
        self.expect(
            "log enable -h circular lldb commands",
            error=True,
            substrs=["the circular buffer handler requires a non-zero buffer size"],
        )

    def test_dump_unsupported(self):
        self.runCmd("log enable lldb commands -f {}".format(self.log_file))
        self.expect(
            "log dump lldb",
            error=True,
            substrs=["log channel 'lldb' does not support dumping"],
        )
