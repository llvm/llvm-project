"""
Tests for the lldb-server Mock Accelerator Plugin.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

"""This will go away once the mock plugin starts creating a target."""
_PLUGIN_OUTPUT_PATH = "/tmp/accelerator_plugin_test.txt"


class MockAcceleratorPluginTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def clean_expected_output_file(self):
        try:
            os.remove(_PLUGIN_OUTPUT_PATH)
        except OSError:
            pass

    def tearDown(self):
        super().tearDown()
        self.clean_expected_output_file()

    @add_test_categories(["llgs"])
    def test_mock_accelerator_plugin_writes_pid(self):
        """
        When LLDB_SERVER_ENABLE_MOCK_ACCELERATOR_PLUGIN is set, the mock accelerator
        plugin must create /tmp/accelerator_plugin_test.txt containing a valid PID.
        """
        self.build()

        # Remove any stale file left by a previous run.
        self.clean_expected_output_file()

        # Launch the target and let it run to completion.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid())

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertEqual(
            process.GetState(),
            lldb.eStateExited,
            "Process should have exited cleanly",
        )

        # After the process exits, verify the plugin output file.
        self.assertTrue(
            os.path.exists(_PLUGIN_OUTPUT_PATH),
            "Mock accelerator plugin did not create %s" % _PLUGIN_OUTPUT_PATH,
        )
        with open(_PLUGIN_OUTPUT_PATH) as f:
            content = f.read().strip()
        self.assertTrue(
            content.isdigit(),
            "Plugin output file should contain an integer PID, got: %r" % content,
        )

        pid = int(content)
        self.assertGreater(
            pid,
            0,
            "PID in plugin output file should be a valid positive integer",
        )
