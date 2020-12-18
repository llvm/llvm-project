"""
Test target commands: target.auto-install-main-executable.
"""

import time
import gdbremote_testcase

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestAutoInstallMainExecutable(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    @llgs_test
    @no_debug_info_test
    @skipIf(remote=False)
    @expectedFailureAll(hostoslist=["windows"], triple='.*-android')
    def test_target_auto_install_main_executable(self):
        self.build()

        # Manually install the modified binary.
        working_dir = lldb.remote_platform.GetWorkingDirectory()
        src_device = lldb.SBFileSpec(self.getBuildArtifact("a.device.out"))
        dest = lldb.SBFileSpec(os.path.join(working_dir, "a.out"))
        err = lldb.remote_platform.Put(src_device, dest)
        if err.Fail():
            raise RuntimeError(
                "Unable copy '%s' to '%s'.\n>>> %s" %
                (src_device.GetFilename(), working_dir, err.GetCString()))

        m = re.search("^(.*)://([^/]*):(.*)$", configuration.lldb_platform_url)
        protocol = m.group(1)
        hostname = m.group(2)
        hostport = int(m.group(3))
        listen_url = "*:"+str(hostport+1)

        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--server"
            ]

        self.spawnSubprocess(
            self.debug_monitor_exe,
            commandline_args,
            install_remote=False)

        # Wait for the new process gets ready.
        time.sleep(0.1)

        self.dbg.SetAsync(False)

        new_platform = lldb.SBPlatform(lldb.remote_platform.GetName())
        self.dbg.SetSelectedPlatform(new_platform)

        connect_url = "%s://%s:%s" % (protocol, hostname, str(hostport+1))

        # Test the default setting.
        self.expect("settings show target.auto-install-main-executable",
                substrs=["target.auto-install-main-executable (boolean) = true"],
                msg="Default settings for target.auto-install-main-executable failed.")

        # Disable the auto install.
        self.runCmd("settings set target.auto-install-main-executable false")
        self.expect("settings show target.auto-install-main-executable",
            substrs=["target.auto-install-main-executable (boolean) = false"])

        self.runCmd("platform select %s"%configuration.lldb_platform_name)
        self.runCmd("platform connect %s" % (connect_url))

        # Create the target with the original file.
        self.runCmd("target create --remote-file %s %s "%
                                        (os.path.join(working_dir,dest.GetFilename()),
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
