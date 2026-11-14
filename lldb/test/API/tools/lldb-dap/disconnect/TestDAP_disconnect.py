"""
Test lldb-dap disconnect request
"""

import os
import subprocess
import uuid

from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import *


@requireNotWasm("no attach support")
class TestDAP_disconnect(DAPTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    source = "main.cpp"

    @skipIfWindows
    def test_launch(self):
        """
        This test launches a process that would creates a file, but we disconnect
        before the file is created, which terminates the process and thus the file is not
        created.
        """
        program = self.getBuildArtifact("a.out")
        side_effect = f"{program}.side_effect"
        session = self.build_and_create_session(disconnect_automatically=False)
        with session.configure(LaunchArgs(program, stopOnEntry=True)) as ctx:
            # We set a breakpoint right before the side effect file is created
            session.resolve_source_breakpoints(
                self.source, [line_number(self.source, "// breakpoint")]
            )
        stop_event = session.verify_stopped_on_entry(after=ctx.process_event)

        # Verify we haven't produced the side effect file yet.
        self.assertFalse(os.path.exists(side_effect))

        # Verify we receive the exited and terminated event before the disconnect response.
        response = session.disconnect(terminateDebuggee=True)
        exited = session.wait_for_exited_event(after=stop_event)
        self.assertNotEqual(exited.body.exitCode, 0)
        terminated = session.wait_for_terminated_event(after=stop_event)
        self.assertLess(terminated.seq, response.seq)
        self.assertLess(exited.seq, response.seq)

        with self.assertRaises(DAPError):
            # We should not receive a second terminated event. The check doesn't
            # stall, since the event history is closed.
            session.wait_for_terminated_event(after=terminated)

        # Verify we didn't produce the side effect file.
        self.assertFalse(os.path.exists(side_effect))

    @skipIfWindows
    @expectedFailureNetBSD
    def test_attach(self):
        """
        This test attaches to a process that creates a file. We attach and disconnect
        before the file is created, and as the process is not terminated upon disconnection,
        the file is created anyway.
        """
        session = self.build_and_create_session(disconnect_automatically=False)
        program = self.getBuildArtifact("a.out")
        side_effect = program + ".side_effect"

        # Use a file as a synchronization point between test and inferior.
        sync_file_path = lldbutil.append_to_process_working_directory(
            self, f"sync_file_{uuid.uuid4().hex}"
        )

        proc = self.spawnSubprocess(
            program, [sync_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        lldbutil.wait_for_file_on_target(self, sync_file_path)

        with session.configure(AttachArgs(pid=proc.pid)) as cm:
            line = line_number(self.source, "// attach breakpoint")
            [bp_id] = session.resolve_source_breakpoints(self.source, [line])
        stop_event = session.verify_stopped_on_breakpoint(bp_id, after=cm.process_event)

        self.assertFalse(os.path.exists(side_effect))

        top_frame = session.top_frame_from(stop_event)
        self.logger.info("frame name: %s", top_frame.name)
        top_frame.evaluate("`expr wait_for_attach = false;", context="repl")

        # Verify the variable changed.
        wait_for_attach_var = top_frame.evaluate("wait_for_attach", context="hover")
        self.assertEqual(wait_for_attach_var.result, "false")

        # Verify we receive the terminated event before the disconnect response.
        disconnect_resp = session.disconnect()
        possible_events = (ExitedEvent, TerminatedEvent, StoppedEvent)
        end_event = session.wait_for_any_event(possible_events, after=stop_event)
        self.assertIsInstance(end_event, TerminatedEvent)
        self.assertLess(end_event.seq, disconnect_resp.seq)

        # Wait for the process to run to completion.
        proc.wait(timeout=10)

        self.assertTrue(os.path.exists(side_effect))

    def test_disconnect_after_exit(self):
        """
        Disconnecting after the process exited doesn't send another
        "terminated" event.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session(disconnect_automatically=False)
        process_event = session.launch(LaunchArgs(program))
        session.verify_process_exited(after=process_event)
        terminated = session.wait_for_terminated_event(after=process_event)

        session.disconnect()
        with self.assertRaises(DAPError):
            # We should not receive a second terminated event. The check doesn't
            # stall, since the event history is closed.
            session.wait_for_terminated_event(after=terminated)
