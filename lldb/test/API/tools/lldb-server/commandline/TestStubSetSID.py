import gdbremote_testcase
import lldbgdbserverutils
import os
import select
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestStubSetSIDTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):

    def get_stub_sid(self, extra_stub_args=None):
        # Launch debugserver
        if extra_stub_args:
            self.debug_monitor_extra_args += extra_stub_args

        server = self.launch_debug_monitor()
        self.assertIsNotNone(server)
        self.assertTrue(
            lldbgdbserverutils.process_is_running(
                server.pid, True))

        # Get the process id for the stub.
        return os.getsid(server.pid)

    @skipIfWindows
    @skipIfRemote  # --setsid not used on remote platform and currently it is also impossible to get the sid of lldb-platform running on a remote target
    def test_sid_is_same_without_setsid(self):
        self.set_inferior_startup_launch()

        stub_sid = self.get_stub_sid()
        self.assertEqual(stub_sid, os.getsid(0))

    @skipIfWindows
    @skipIfRemote  # --setsid not used on remote platform and currently it is also impossible to get the sid of lldb-platform running on a remote target
    def test_sid_is_different_with_setsid(self):
        self.set_inferior_startup_launch()

        stub_sid = self.get_stub_sid(["--setsid"])
        self.assertNotEqual(stub_sid, os.getsid(0))

    @skipIfWindows
    @skipIfRemote  # --setsid not used on remote platform and currently it is also impossible to get the sid of lldb-platform running on a remote target
    def test_sid_is_different_with_S_llgs(self):
        self.set_inferior_startup_launch()

        stub_sid = self.get_stub_sid(["-S"])
        self.assertNotEqual(stub_sid, os.getsid(0))
