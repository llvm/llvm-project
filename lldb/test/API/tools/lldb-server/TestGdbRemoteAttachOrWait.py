
import os
from time import sleep

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteAttachOrWait(gdbremote_testcase.GdbRemoteTestCaseBase):

    @skipIfWindows # This test is flaky on Windows
    def test_launch_before_attach_with_vAttachOrWait(self):
        exe = '%s_%d' % (self.testMethodName, os.getpid())
        self.build(dictionary={'EXE': exe})
        self.set_inferior_startup_attach_manually()

        # Start the inferior, start the debug monitor, nothing is attached yet.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["sleep:60"],
            inferior_exe_path=self.getBuildArtifact(exe))
        self.assertIsNotNone(procs)

        # Make sure the target process has been launched.
        inferior = procs.get("inferior")
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        self.assertTrue(
            lldbgdbserverutils.process_is_running(
                inferior.pid, True))

        # Add attach packets.
        self.test_sequence.add_log_lines([
            # Do the attach.
            "read packet: $vAttachOrWait;{}#00".format(lldbgdbserverutils.gdbremote_hex_encode_string(exe)),
            # Expect a stop notification from the attach.
            {"direction": "send",
             "regex": r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$",
             "capture": {1: "stop_signal_hex"}},
        ], True)
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)

    @skipIfWindows # This test is flaky on Windows
    def test_launch_after_attach_with_vAttachOrWait(self):
        exe = '%s_%d' % (self.testMethodName, os.getpid())
        self.build(dictionary={'EXE': exe})
        self.set_inferior_startup_attach_manually()

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        self.do_handshake()
        self.test_sequence.add_log_lines([
            # Do the attach.
            "read packet: $vAttachOrWait;{}#00".format(lldbgdbserverutils.gdbremote_hex_encode_string(exe)),
        ], True)
        # Run the stream until attachWait.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Sleep so we're sure that the inferior is launched after we ask for the attach.
        sleep(1)

        # Launch the inferior.
        inferior = self.launch_process_for_attach(
            inferior_args=["sleep:60"],
            exe_path=self.getBuildArtifact(exe))
        self.assertIsNotNone(inferior)
        self.assertTrue(inferior.pid > 0)
        self.assertTrue(
            lldbgdbserverutils.process_is_running(
                inferior.pid, True))

        # Make sure the attach succeeded.
        self.test_sequence.add_log_lines([
            {"direction": "send",
             "regex": r"^\$T([0-9a-fA-F]{2})[^#]*#[0-9a-fA-F]{2}$",
             "capture": {1: "stop_signal_hex"}},
        ], True)
        self.add_process_info_collection_packets()


        # Run the stream sending the response..
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response.
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, inferior.pid)
