import os
from time import sleep

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteAttachWait(gdbremote_testcase.GdbRemoteTestCaseBase):
    def _set_up_inferior(self):
        self._exe_to_attach = "%s_%d" % (self.testMethodName, os.getpid())
        self.build(dictionary={"EXE": self._exe_to_attach, "CXX_SOURCES": "main.cpp"})

        if self.getPlatform() != "windows":
            # Use a shim to ensure that the process is ready to be attached from
            # the get-go.
            self._exe_to_run = "shim"
            self._run_args = [self.getBuildArtifact(self._exe_to_attach)]
            self.build(dictionary={"EXE": self._exe_to_run, "CXX_SOURCES": "shim.cpp"})
        else:
            self._exe_to_run = self._exe_to_attach
            self._run_args = []

    def _launch_inferior(self, args):
        inferior = self.spawnSubprocess(self.getBuildArtifact(self._exe_to_run), args)
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        self.assertTrue(lldbgdbserverutils.process_is_running(inferior.pid, True))
        return inferior

    def _launch_and_wait_for_init(self):
        sync_file_path = lldbutil.append_to_process_working_directory(
            self, "process_ready"
        )
        inferior = self._launch_inferior(self._run_args + [sync_file_path])
        lldbutil.wait_for_file_on_target(self, sync_file_path)
        return inferior

    def _attach_packet(self, packet_type):
        return "read packet: ${};{}#00".format(
            packet_type,
            lldbgdbserverutils.gdbremote_hex_encode_string(self._exe_to_attach),
        )

    @skipIfWindows  # This test is flaky on Windows
    def test_attach_with_vAttachWait(self):
        self._set_up_inferior()

        self.set_inferior_startup_attach_manually()
        server = self.connect_to_debug_monitor()
        self.do_handshake()

        # Launch the first inferior (we shouldn't attach to this one).
        self._launch_and_wait_for_init()

        self.test_sequence.add_log_lines([self._attach_packet("vAttachWait")], True)
        # Run the stream until attachWait.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Sleep so we're sure that the inferior is launched after we ask for the attach.
        sleep(1)

        # Launch the second inferior (we SHOULD attach to this one).
        inferior_to_attach = self._launch_inferior(self._run_args)

        # Make sure the attach succeeded.
        self.test_sequence.add_log_lines(
            [
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$",
                    "capture": {1: "stop_signal_hex"},
                },
            ],
            True,
        )
        self.add_process_info_collection_packets()

        # Run the stream sending the response..
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response.
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get("pid", None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior_to_attach.pid)

    @skipIfWindows  # This test is flaky on Windows
    def test_launch_before_attach_with_vAttachOrWait(self):
        self._set_up_inferior()

        self.set_inferior_startup_attach_manually()
        server = self.connect_to_debug_monitor()
        self.do_handshake()

        inferior = self._launch_and_wait_for_init()

        # Add attach packets.
        self.test_sequence.add_log_lines(
            [
                # Do the attach.
                self._attach_packet("vAttachOrWait"),
                # Expect a stop notification from the attach.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$",
                    "capture": {1: "stop_signal_hex"},
                },
            ],
            True,
        )
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get("pid", None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)

    @skipIfWindows  # This test is flaky on Windows
    def test_launch_after_attach_with_vAttachOrWait(self):
        self._set_up_inferior()

        self.set_inferior_startup_attach_manually()
        server = self.connect_to_debug_monitor()
        self.do_handshake()

        self.test_sequence.add_log_lines([self._attach_packet("vAttachOrWait")], True)
        # Run the stream until attachWait.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Sleep so we're sure that the inferior is launched after we ask for the attach.
        sleep(1)

        # Launch the inferior.
        inferior = self._launch_inferior(self._run_args)

        # Make sure the attach succeeded.
        self.test_sequence.add_log_lines(
            [
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$",
                    "capture": {1: "stop_signal_hex"},
                },
            ],
            True,
        )
        self.add_process_info_collection_packets()

        # Run the stream sending the response..
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response.
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get("pid", None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)
