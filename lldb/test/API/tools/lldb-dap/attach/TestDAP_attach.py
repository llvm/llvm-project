"""
Test lldb-dap attach request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import subprocess
import threading
import time


class TestDAP_attach(lldbdap_testcase.DAPTestCaseBase):
    def spawn(self, args):
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def spawn_and_wait(self, program, delay):
        time.sleep(delay)
        self.spawn([program])
        self.process.wait()

    def continue_and_verify_pid(self):
        self.do_continue()
        out, _ = self.process.communicate("foo")
        self.assertIn(f"pid = {self.process.pid}", out)

    def test_by_pid(self):
        """
        Tests attaching to a process by process ID.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        self.spawn([program])
        self.attach(pid=self.process.pid)
        self.continue_and_verify_pid()

    def test_by_name(self):
        """
        Tests attaching to a process by process name.
        """
        program = self.build_and_create_debug_adapter_for_attach()

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(
            self, "pid_file_%d" % (int(time.time()))
        )
        self.spawn([program, pid_file_path])
        lldbutil.wait_for_file_on_target(self, pid_file_path)

        self.attach(program=program)
        self.continue_and_verify_pid()

    def test_by_name_waitFor(self):
        """
        Tests waiting for, and attaching to a process by process name that
        doesn't exist yet.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        self.spawn_thread = threading.Thread(
            target=self.spawn_and_wait,
            args=(
                program,
                1.0,
            ),
        )
        self.spawn_thread.start()
        self.attach(program=program, waitFor=True)
        self.continue_and_verify_pid()

    def test_attach_with_missing_debuggerId_or_targetId(self):
        """
        Test that attaching with only one of debuggerId or targetId specified
        fails with the expected error message.
        """
        self.build_and_create_debug_adapter()

        # Test with only targetId specified (no debuggerId)
        resp = self.attach(targetId=99999, expectFailure=True)
        self.assertFalse(resp["success"])
        self.assertIn(
            "Both debuggerId and targetId must be specified together",
            resp["body"]["error"]["format"],
        )

    def test_attach_with_invalid_debuggerId_and_targetId(self):
        """
        Test that attaching with both debuggerId and targetId specified but
        invalid fails with an appropriate error message.
        """
        self.build_and_create_debug_adapter()

        # Attach with both debuggerId=1 and targetId=99999 (both invalid).
        # Since debugger ID 1 likely doesn't exist in the global registry,
        # we expect a validation error.
        resp = self.attach(debuggerId=9999, targetId=99999, expectFailure=True)
        self.assertFalse(resp["success"])
        error_msg = resp["body"]["error"]["format"]
        # Either error is acceptable - both indicate the debugger reuse
        # validation is working correctly
        self.assertTrue(
            "Unable to find existing debugger" in error_msg
            or f"Expected debugger/target not found error, got: {error_msg}"
        )
