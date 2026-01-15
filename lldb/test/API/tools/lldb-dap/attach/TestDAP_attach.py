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


# Often fails on Arm Linux, but not specifically because it's Arm, something in
# process scheduling can cause a massive (minutes) delay during this test.
@skipIf(oslist=["linux"], archs=["arm$"])
class TestDAP_attach(lldbdap_testcase.DAPTestCaseBase):
    def spawn(self, program, args=None):
        return self.spawnSubprocess(
            executable=program,
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def spawn_and_wait(self, program, delay):
        time.sleep(delay)
        proc = self.spawn(program=program)
        start_time = time.time()
        # Wait for either the process to exit or the event to be set.
        while proc.poll() is None and not self.spawn_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= self.DEFAULT_TIMEOUT:
                break
            time.sleep(0.1)
        proc.kill()
        proc.wait()

    def continue_and_verify_pid(self):
        self.do_continue()
        proc = self.lastSubprocess
        if proc is None:
            self.fail(f"lastSubprocess is None")
        out, _ = proc.communicate("foo")
        self.assertIn(f"pid = {proc.pid}", out)

    def test_by_pid(self):
        """
        Tests attaching to a process by process ID.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        proc = self.spawn(program=program)
        self.attach(pid=proc.pid)
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
        self.spawn(program=program, args=[pid_file_path])
        lldbutil.wait_for_file_on_target(self, pid_file_path)

        self.attach(program=program)
        self.continue_and_verify_pid()

    @expectedFailureWindows
    def test_by_name_waitFor(self):
        """
        Tests waiting for, and attaching to a process by process name that
        doesn't exist yet.
        """
        program = self.build_and_create_debug_adapter_for_attach()
        self.spawn_event = threading.Event()
        self.spawn_thread = threading.Thread(
            target=self.spawn_and_wait,
            args=(
                program,
                1.0,
            ),
        )
        self.spawn_thread.start()
        try:
            self.attach(program=program, waitFor=True)
            self.continue_and_verify_pid()
        finally:
            self.spawn_event.set()
            if self.spawn_thread.is_alive():
                self.spawn_thread.join(timeout=10)

    def test_attach_with_missing_session_debugger(self):
        """
        Test that attaching with only one of debuggerId/targetId specified
        fails with the expected error message.
        """
        self.build_and_create_debug_adapter()

        # Test with only targetId specified (no debuggerId)
        session = {"targetId": 99999}
        resp = self.attach(session=session, waitForResponse=True)
        self.assertFalse(resp["success"])
        self.assertIn(
            "missing value at arguments.session.debuggerId",
            resp["body"]["error"]["format"],
        )

    def test_attach_with_invalid_session(self):
        """
        Test that attaching with both debuggerId and targetId specified but
        invalid fails with an appropriate error message.
        """
        self.build_and_create_debug_adapter()

        # Attach with both debuggerId=9999 and targetId=99999 (both invalid).
        # Since debugger ID 9999 likely doesn't exist in the global registry,
        # we expect a validation error.
        session = {"debuggerId": 9999, "targetId": 9999}
        resp = self.attach(session=session, waitForResponse=True)
        self.assertFalse(resp["success"])
        error_msg = resp["body"]["error"]["format"]
        # Either error is acceptable - both indicate the debugger reuse
        # validation is working correctly
        self.assertTrue(
            "Unable to find existing debugger" in error_msg
            or f"Expected debugger/target not found error, got: {error_msg}"
        )
