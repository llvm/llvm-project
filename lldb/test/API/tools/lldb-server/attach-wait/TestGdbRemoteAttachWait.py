
import os
from time import sleep

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteAttachWait(gdbremote_testcase.GdbRemoteTestCaseBase):

    @skipIfWindows # This test is flaky on Windows
    def test_attach_with_vAttachWait(self):
        exe = '%s_%d' % (self.testMethodName, os.getpid())
        exe_to_attach = exe
        sync_file_path = lldbutil.append_to_process_working_directory(self, "process_ready")
        args = [sync_file_path]

        def launch_inferior():
            inferior = self.launch_process_for_attach(
                inferior_args=args,
                exe_path=self.getBuildArtifact(exe))
            self.assertIsNotNone(inferior)
            self.assertTrue(inferior.pid > 0)
            self.assertTrue(
                lldbgdbserverutils.process_is_running(
                    inferior.pid, True))
            return inferior

        self.build(dictionary={'EXE': exe, 'CXX_SOURCES': 'main.cpp'})
        if self.getPlatform() != "windows":
            # Use a shim to ensure that the process is ready to be attached from
            # the get-go.
            args = [self.getBuildArtifact(exe)] + args
            exe = "shim"
            self.build(dictionary={'EXE': exe, 'CXX_SOURCES': 'shim.cpp'})

        self.set_inferior_startup_attach_manually()

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        # Launch the first inferior (we shouldn't attach to this one).
        launch_inferior()

        lldbutil.wait_for_file_on_target(self, sync_file_path)

        self.do_handshake()
        self.test_sequence.add_log_lines([
            # Do the attach.
            "read packet: $vAttachWait;{}#00".format(
                lldbgdbserverutils.gdbremote_hex_encode_string(exe_to_attach)),
        ], True)
        # Run the stream until attachWait.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Sleep so we're sure that the inferior is launched after we ask for the attach.
        sleep(1)

        # Launch the second inferior (we SHOULD attach to this one).
        inferior_to_attach = launch_inferior()

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
        self.assertEqual(reported_pid, inferior_to_attach.pid)
