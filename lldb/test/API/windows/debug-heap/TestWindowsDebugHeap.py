"""
Test that LLDB disables the debug heap on Windows.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from typing import List


@skipUnlessWindows
class DebugHeapTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def tearDown(self):
        self.runCmd("settings clear platform.plugin.windows.disable-debug-heap")
        return super().tearDown()

    def _run_to_exit(self, envp: List[str]=[]):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.dbg.SetAsync(False)
        process = target.LaunchSimple([], envp, self.get_process_working_directory())
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateExited)
        return process.GetSTDOUT(256)

    def test_default(self):
        output = self._run_to_exit()
        self.assertIn("_NO_DEBUG_HEAP=1", output)
        output = self._run_to_exit(["_NO_DEBUG_HEAP=2"])
        self.assertIn("_NO_DEBUG_HEAP=2", output)

    def test_disabled(self):
        self.runCmd("settings set platform.plugin.windows.disable-debug-heap false")
        output = self._run_to_exit()
        self.assertNotIn("_NO_DEBUG_HEAP", output)
        output = self._run_to_exit(["_NO_DEBUG_HEAP=2"])
        self.assertIn("_NO_DEBUG_HEAP=2", output)

    def test_enabled(self):
        self.runCmd("settings set platform.plugin.windows.disable-debug-heap true")
        output = self._run_to_exit()
        self.assertIn("_NO_DEBUG_HEAP=1", output)
        output = self._run_to_exit(["_NO_DEBUG_HEAP=2"])
        self.assertIn("_NO_DEBUG_HEAP=2", output)
