"""
Tests the exit code/description coming from the debugserver.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    @skipUnlessDarwin
    @skipIfOutOfTreeDebugserver
    def test_abort(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        process = target.LaunchSimple(None, None, None)
        # Continue until process is terminated.
        process.Continue()
        # Test for the abort signal code.
        self.assertEqual(process.GetExitStatus(), 6)
        # Test for the exit code description.
        self.assertEqual(process.GetExitDescription(), "Terminated due to signal 6")
