import socket
import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPlatformProcessConnect(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @expectedFailureAll(hostoslist=["windows"], triple=".*-android")
    @skipIfDarwin  # lldb-server not found correctly
    @expectedFailureAll(oslist=["windows"])  # process modules not loaded
    @add_test_categories(["lldb-server"])
    def test_platform_process_connect(self):
        self.build()

        hostname = socket.getaddrinfo("localhost", 0, proto=socket.IPPROTO_TCP)[0][4][0]
        listen_url = "[%s]:0" % hostname

        port_file = self.getBuildArtifact("port")
        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--socket-file",
            port_file,
            "--",
            self.getBuildArtifact("a.out"),
            "foo",
        ]
        self.spawnSubprocess(lldbgdbserverutils.get_lldb_server_exe(), commandline_args)

        socket_id = lldbutil.wait_for_file_on_target(self, port_file)

        new_platform = lldb.SBPlatform("remote-" + self.getPlatform())
        self.dbg.SetSelectedPlatform(new_platform)

        connect_url = "connect://[%s]:%s" % (hostname, socket_id)
        self.runCmd("platform connect %s" % connect_url)

        lldbutil.run_break_set_by_symbol(self, "main")
        process = self.process()

        process.Continue()

        frame = self.frame()
        self.assertEqual(frame.GetFunction().GetName(), "main")
        self.assertEqual(frame.FindVariable("argc").GetValueAsSigned(), 2)
        process.Continue()
