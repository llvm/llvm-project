"""
Tests for Windows ConPTY (Pseudo Console) process I/O.

These tests explicitly exercise the ConPTY path by clearing
LLDB_LAUNCH_FLAG_USE_PIPES, which the test suite sets globally to avoid
ConPTY VT-sequence pollution in unrelated tests.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

# Must match main.c.
_NUM_LINES = 500


class ConPTYTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        import os

        TestBase.setUp(self)
        # Clear LLDB_LAUNCH_FLAG_USE_PIPES so LLDB uses ConPTY instead of
        # anonymous pipes. Restored in tearDown.
        self._saved_pipes_flag = os.environ.pop("LLDB_LAUNCH_FLAG_USE_PIPES", None)

    def tearDown(self):
        import os

        if self._saved_pipes_flag is not None:
            os.environ["LLDB_LAUNCH_FLAG_USE_PIPES"] = self._saved_pipes_flag
        TestBase.tearDown(self)

    @staticmethod
    def _strip_output(text: str) -> str:
        """
        Strip VT sequences that ConPTY injects around the inferior's output
        (CSI sequences like SGR resets, mode switches, cursor queries; and
        OSC sequences like window-title sets) so the assertion only checks
        the inferior's actual stdout content.
        """
        import re

        return re.sub(r"\x1b(?:\[[0-9;?]*[A-Za-z]|\][^\x07]*\x07)", "", text)

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
    @skipUnlessWindowsConPTY2022
    @skipIf(oslist=["windows"], archs=["aarch64"], bugnumber="#194069")
    def test_stdout_delivery(self):
        """ConPTY delivers the inferior's stdout to LLDB."""
        import re

        output = self._run_to_exit("basic")
        output = self._strip_output(output)
        self.assertIn("Hello from ConPTY\r\n", output)

    @skipUnlessWindows
    @skipUnlessWindowsConPTY2022
    @skipIf(oslist=["windows"], archs=["aarch64"], bugnumber="#194069")
    def test_large_output(self):
        """ConPTY delivers all output lines when output spans multiple reads."""
        import re

        output = self._run_to_exit("large")
        output = self._strip_output(output)
        output_lines = output.split("\r\n")[:-1]

        self.assertEqual(
            _NUM_LINES, len(output_lines), "Got fewer lines than expected."
        )

        for i, line in enumerate(output_lines):
            self.assertEqual("line %04d" % i, line)

    @skipUnlessWindows
    @skipUnlessWindowsConPTY
    @skipIf(oslist=["windows"], archs=["aarch64"], bugnumber="#194069")
    def test_basic_output_without_vt_check(self):
        """ConPTY delivers the inferior's stdout on all supported Windows versions.

        Unlike test_stdout_delivery, this test strips VT escape sequences before
        asserting, so it passes on older Windows versions (e.g. Windows Server
        2019) where ConPTY emits different sequences.
        """

        import re

        output = self._run_to_exit("basic")
        stripped = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output)
        self.assertIn("Hello from ConPTY", stripped)

    @skipUnlessWindows
    @skipUnlessWindowsConPTY2022
    @skipIf(oslist=["windows"], archs=["aarch64"], bugnumber="#194069")
    def test_no_screen_clear_on_init(self):
        """PSEUDOCONSOLE_INHERIT_CURSOR prevents ConPTY from emitting
        screen-clearing sequences that would overwrite existing terminal output.

        With PSEUDOCONSOLE_INHERIT_CURSOR, ConPTY queries the current cursor
        position (ESC[6n) and skips the full-screen reset it would otherwise
        emit.  Verify that none of those reset sequences appear in the process
        output.
        """
        output = self._run_to_exit("basic")

        # Emitted by ConPTY during a full-screen init (no cursor inheritance).
        self.assertNotIn("\x1b[2J", output)  # clear screen
        self.assertNotIn("\x1b[3J", output)  # erase scrollback
        self.assertNotIn("\x1b[H", output)  # cursor home
