"""
Test target commands: target.auto-install-main-executable.
"""

import socket
import time
import lldbgdbserverutils

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestAutoInstallMainExecutable(TestBase):
    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @expectedFailureAll(oslist=["windows"]) # process modules not loaded
    def test_target_auto_install_main_executable(self):
        self.build()

        hostname = socket.getaddrinfo("localhost", 0, proto=socket.IPPROTO_TCP)[0][4][0]
        listen_url = "[%s]:0"%hostname

        port_file = self.getBuildArtifact("port")
        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--socket-file",
            port_file]
        self.spawnSubprocess(
            lldbgdbserverutils.get_lldb_server_exe(),
            commandline_args)

        socket_id = lldbutil.wait_for_file_on_target(self, port_file)

        new_platform = lldb.SBPlatform("remote-" + self.getPlatform())
        self.dbg.SetSelectedPlatform(new_platform)

        connect_url = "connect://[%s]:%s" % (hostname, socket_id)
        connect_opts = lldb.SBPlatformConnectOptions(connect_url)
        self.assertSuccess(new_platform.ConnectRemote(connect_opts))

        wd = self.getBuildArtifact("wd")
        os.mkdir(wd)
        new_platform.SetWorkingDirectory(wd)


        # Manually install the modified binary.
        src_device = lldb.SBFileSpec(self.getBuildArtifact("a.device.out"))
        dest = lldb.SBFileSpec(os.path.join(wd, "a.out"))
        self.assertSuccess(new_platform.Put(src_device, dest))

        # Test the default setting.
        self.expect("settings show target.auto-install-main-executable",
                substrs=["target.auto-install-main-executable (boolean) = true"],
                msg="Default settings for target.auto-install-main-executable failed.")

        # Disable the auto install.
        self.runCmd("settings set target.auto-install-main-executable false")
        self.expect("settings show target.auto-install-main-executable",
            substrs=["target.auto-install-main-executable (boolean) = false"])

        # Create the target with the original file.
        self.runCmd("target create --remote-file %s %s "%
                                        (dest.fullpath,
                                            self.getBuildArtifact("a.out")))

        target = self.dbg.GetSelectedTarget()
        breakpoint = target.BreakpointCreateByName("main")

        launch_info = target.GetLaunchInfo()
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunction().GetName(), "main")

        self.expect("target variable build", substrs=['"device"'],
                msg="Magic in the binary is wrong")

        process.Continue()
