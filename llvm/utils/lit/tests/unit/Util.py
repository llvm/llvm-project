# RUN: %{python} %s
# UNSUPPORTED: system-windows

import unittest
import platform
import time

from lit.util import runCommandCached
from lit.LitConfig import LitConfig


class TestCommandCache(unittest.TestCase):
    @staticmethod
    def _lit_config():
        return LitConfig(
            progname="lit",
            path=[],
            diagnostic_level="note",
            useValgrind=False,
            valgrindLeakCheck=False,
            valgrindArgs=[],
            noExecute=False,
            debug=False,
            isWindows=(platform.system() == "Windows"),
            order="smart",
            params={},
        )

    def test_basic(self):
        lit_config = self._lit_config()

        self.assertEqual(lit_config.run_command_cached(["echo", "-n", "hi"]), b"hi")
        self.assertNotEqual(lit_config.run_command_cached("ls"), None)

        # Test that arguments (e.g. text=True) get forwarded to subprocess.run
        self.assertEqual(
            lit_config.run_command_cached(["echo", "-n", "hi"], text=True), "hi"
        )

        # shell=True is not implied
        self.assertEqual(
            lit_config.run_command_cached("ls -al", allow_failure=True), None
        )
        self.assertNotEqual(
            lit_config.run_command_cached("ls -al", allow_failure=True, shell=True),
            None,
        )

        self.assertEqual(
            lit_config.run_command_cached("exit 0", shell=True, allow_failure=True), b""
        )

    def test_fatal(self):
        lit_config = self._lit_config()

        # Test fatal errors
        fatal_counter = 0

        def wrap_fatal(msg):
            nonlocal fatal_counter
            fatal_counter += 1

        lit_config.fatal = wrap_fatal
        lit_config.run_command_cached(["asdfghjkl"])
        self.assertEqual(fatal_counter, 1)

        self.assertEqual(
            lit_config.run_command_cached(["asdfghjkl"], allow_failure=True), None
        )
        self.assertEqual(
            lit_config.run_command_cached("exit 1", shell=True, allow_failure=True),
            None,
        )
        self.assertEqual(fatal_counter, 1)

    def test_cache(self):
        lit_config = self._lit_config()

        # Check the date (with nanoseconds)
        date = lit_config.run_command_cached("date -Ins", shell=True)
        self.assertNotEqual(date, None)

        # Second time should be cached, i.e. equal to the first
        self.assertEqual(lit_config.run_command_cached("date -Ins", shell=True), date)


if __name__ == "__main__":
    unittest.main()
