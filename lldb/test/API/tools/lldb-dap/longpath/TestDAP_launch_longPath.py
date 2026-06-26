"""
Test that lldb-dap reports the full executable path in the "process" event when
the program lives at a path longer than the Windows MAX_PATH limit (260).
"""

import os
import shutil

import lldbdap_testcase
from lldbsuite.test.decorators import *

MAX_PATH = 260


@skipUnlessWindows
class TestDAP_launch_longPath(lldbdap_testcase.DAPTestCaseBase):
    def _long_path(self, path):
        return "\\\\?\\" + os.path.abspath(path)

    def _normalize(self, path):
        if path.startswith("\\\\?\\"):
            path = path[4:]
        return os.path.normcase(os.path.normpath(path))

    def _make_long_dir(self):
        components = [self.getBuildArtifact("deep")] + ["d" * 80] * 3
        target_dir = os.path.join(*components)
        try:
            os.makedirs(self._long_path(target_dir), exist_ok=True)
        except OSError:
            return None
        return target_dir

    def test_process_event_long_path(self):
        self.build()
        program = self.getBuildArtifact("a.out")

        long_dir = self._make_long_dir()
        if long_dir is None:
            self.skipTest("OS cannot create paths longer than MAX_PATH")

        long_program = os.path.join(long_dir, os.path.basename(program))
        shutil.copyfile(program, self._long_path(long_program))
        self.assertGreater(len(os.path.abspath(long_program)), MAX_PATH)

        self.create_debug_adapter()
        self.launch_and_configurationDone(long_program)

        process_event = self.dap_server.wait_for_event(["process"])
        self.assertIsNotNone(process_event, "lldb-dap sent a process event")
        name = process_event["body"]["name"]
        self.assertGreater(
            len(name), MAX_PATH, "process event name must not be truncated"
        )
        self.assertEqual(
            self._normalize(name),
            self._normalize(long_program),
        )

        self.dap_server.wait_for_event(["terminated", "exited"])
