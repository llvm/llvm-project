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

    def setUp(self):
        super(TestAutoInstallMainExecutable, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(TestAutoInstallMainExecutable, self).tearDown()

    @llgs_test
    @no_debug_info_test
    @skipIf(remote=False)
    @expectedFailureAll(hostoslist=["windows"], triple='.*-android')
    def test_target_auto_install_main_executable(self):
        self.build()
        self.init_llgs_test(False)

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
        self.addTearDownHook(self.cleanupSubprocesses)

        # Wait for the new process gets ready.
        time.sleep(0.1)

        new_debugger = lldb.SBDebugger.Create()
        new_debugger.SetAsync(False)

        def del_debugger(new_debugger=new_debugger):
            del new_debugger
        self.addTearDownHook(del_debugger)

        new_platform = lldb.SBPlatform(lldb.remote_platform.GetName())
        new_debugger.SetSelectedPlatform(new_platform)
        new_interpreter = new_debugger.GetCommandInterpreter()

        connect_url = "%s://%s:%s" % (protocol, hostname, str(hostport+1))

        command = "platform connect %s" % (connect_url)

        result = lldb.SBCommandReturnObject()

        # Test the default setting.
        new_interpreter.HandleCommand("settings show target.auto-install-main-executable", result)
        self.assertTrue(
            result.Succeeded() and
            "target.auto-install-main-executable (boolean) = true" in result.GetOutput(),
            "Default settings for target.auto-install-main-executable failed.: %s - %s" %
            (result.GetOutput(), result.GetError()))

        # Disable the auto install.
        new_interpreter.HandleCommand("settings set target.auto-install-main-executable false", result)
        new_interpreter.HandleCommand("settings show target.auto-install-main-executable", result)
        self.assertTrue(
            result.Succeeded() and
            "target.auto-install-main-executable (boolean) = false" in result.GetOutput(),
            "Default settings for target.auto-install-main-executable failed.: %s - %s" %
            (result.GetOutput(), result.GetError()))

        new_interpreter.HandleCommand("platform select %s"%configuration.lldb_platform_name, result)
        new_interpreter.HandleCommand(command, result)

        self.assertTrue(
            result.Succeeded(),
            "platform process connect failed: %s - %s" %
            (result.GetOutput(),result.GetError()))

        # Create the target with the original file.
        new_interpreter.HandleCommand("target create --remote-file %s %s "%
                                        (os.path.join(working_dir,dest.GetFilename()), self.getBuildArtifact("a.out")),
                                      result)
        self.assertTrue(
            result.Succeeded(),
            "platform create failed: %s - %s" %
            (result.GetOutput(),result.GetError()))

        target = new_debugger.GetSelectedTarget()
        breakpoint = target.BreakpointCreateByName("main")

        launch_info = lldb.SBLaunchInfo(None)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunction().GetName(), "main")

        new_interpreter.HandleCommand("target variable build", result)
        self.assertTrue(
            result.Succeeded() and
            '"device"' in result.GetOutput(),
            "Magic in the binary is wrong: %s " % result.GetOutput())

        process.Continue()
