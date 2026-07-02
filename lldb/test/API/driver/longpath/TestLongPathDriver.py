"""
Test that the lldb driver can target and run an executable whose path exceeds
the Windows MAX_PATH limit (260 characters).
"""

import os
import shutil
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

MAX_PATH = 260


@skipUnlessWindows
class DriverLongPathTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _long_path(self, path):
        return "\\\\?\\" + os.path.abspath(path)

    def _make_long_dir(self):
        components = [self.getBuildArtifact("deep")] + ["d" * 80] * 3
        target_dir = os.path.join(*components)
        try:
            os.makedirs(self._long_path(target_dir), exist_ok=True)
        except OSError:
            return None
        return target_dir

    def test_driver_runs_long_path_target(self):
        self.build()
        src_exe = self.getBuildArtifact("a.out")

        long_dir = self._make_long_dir()
        if long_dir is None:
            self.skipTest("OS cannot create paths longer than MAX_PATH")

        long_exe = os.path.join(long_dir, os.path.basename(src_exe))
        shutil.copyfile(src_exe, self._long_path(long_exe))
        self.assertGreater(len(os.path.abspath(long_exe)), MAX_PATH)

        # Drive the real lldb executable in batch mode: set a breakpoint, run to
        # it, and list the modules. This both launches an executable past
        # MAX_PATH and prints its full path back.
        proc = subprocess.run(
            [
                lldbtest_config.lldbExec,
                "--batch",
                "--no-lldbinit",
                "-o",
                "breakpoint set --name main",
                "-o",
                "run",
                "-o",
                "image list",
                long_exe,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        output = proc.stdout.decode("utf-8", errors="replace")

        self.assertIn(
            "stop reason = breakpoint",
            output,
            "the driver should launch the long-path target and hit main:\n" + output,
        )
        # The long directory component must appear untruncated in the output
        self.assertIn(
            "d" * 80, output, "the full long path must be reported:\n" + output
        )
