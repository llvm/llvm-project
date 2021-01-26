import socket
import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestPlatformProcessConnect(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    @expectedFailureAll(hostoslist=["windows"], triple='.*-android')
    @skipIfWindows # lldb-server does not terminate correctly
    @skipIfDarwin # lldb-server not found correctly
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])  # Fails randomly
    def test_platform_process_connect(self):
        self.build()

        hostname = socket.getaddrinfo("localhost", 0, proto=socket.IPPROTO_TCP)[0][4][0]
        listen_url = "[%s]:0"%hostname

        port_file = self.getBuildArtifact("port")
        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--socket-file",
            port_file,
            "--",
            self.getBuildArtifact("a.out"),
            "foo"]
        self.spawnSubprocess(
            lldbgdbserverutils.get_lldb_server_exe(),
            commandline_args)

        socket_id = lldbutil.wait_for_file_on_target(self, port_file)

        self.dbg.SetAsync(False)
        new_platform = lldb.SBPlatform("remote-" + self.getPlatform())
        self.dbg.SetSelectedPlatform(new_platform)

        connect_url = "connect://[%s]:%s" % (hostname, socket_id)

        command = "platform connect %s" % (connect_url)
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(command, result)
        self.assertTrue(
            result.Succeeded(),
            "platform process connect failed: %s" %
            result.GetError())

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)

        breakpoint = target.BreakpointCreateByName("main")
        process.Continue()

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunction().GetName(), "main")
        self.assertEqual(frame.FindVariable("argc").GetValueAsSigned(), 2)
        process.Continue()
