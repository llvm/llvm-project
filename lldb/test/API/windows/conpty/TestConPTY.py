"""
Tests for Windows ConPTY (Pseudo Console) process I/O.

These tests explicitly exercise the ConPTY path by clearing
LLDB_LAUNCH_FLAG_USE_PIPES, which the test suite sets globally to avoid
ConPTY VT-sequence pollution in unrelated tests.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

# Must match main.c.
_NUM_LINES = 500


class ConPTYTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        # Clear LLDB_LAUNCH_FLAG_USE_PIPES so LLDB uses ConPTY instead of
        # anonymous pipes. Restored in tearDown.
        self._saved_pipes_flag = os.environ.pop("LLDB_LAUNCH_FLAG_USE_PIPES", None)

    def tearDown(self):
        if self._saved_pipes_flag is not None:
            os.environ["LLDB_LAUNCH_FLAG_USE_PIPES"] = self._saved_pipes_flag
        TestBase.tearDown(self)

    def _run_to_exit(self, mode):
        """Build, launch with *mode* as argv[1], run to exit, return stdout."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.dbg.SetAsync(False)

        process = target.LaunchSimple(
            [mode], None, self.get_process_working_directory()
        )
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)

        self.assertState(process.GetState(), lldb.eStateExited)

        return process.GetSTDOUT(1 << 20)

    @skipUnlessWindows
    def test_stdout_delivery(self):
        """ConPTY delivers the inferior's stdout to LLDB."""
        output = self._run_to_exit("basic")
        self.assertEqual("Hello from ConPTY\r\n", output)

    @skipUnlessWindows
    def test_vt_init_stripped(self):
        """ConPTY VT initialization sequences are stripped from GetSTDOUT."""
        # Sequences emitted by conhost.exe at attach time, defined in
        # ConnectionConPTYWindows.cpp :: StripConPTYInitSequences.
        VT_INIT = "\x1b[?9001l\x1b[?1004l"

        output = self._run_to_exit("basic")

        self.assertIn("Hello from ConPTY\r\n", output)
        self.assertNotIn(VT_INIT, output)

    @skipUnlessWindows
    def test_large_output(self):
        """ConPTY delivers all output lines when output spans multiple reads."""
        output = self._run_to_exit("large")
        output_lines = output.split("\r\n")[:-1]

        self.assertEqual(
            _NUM_LINES, len(output_lines), "Got fewer lines than expected."
        )

        for i, line in enumerate(output_lines):
            self.assertEqual("line %04d" % i, line)
