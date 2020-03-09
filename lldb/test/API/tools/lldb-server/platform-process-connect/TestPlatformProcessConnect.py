
import time

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPlatformProcessConnect(gdbremote_testcase.GdbRemoteTestCaseBase):
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        super(TestPlatformProcessConnect, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(TestPlatformProcessConnect, self).tearDown()

    @llgs_test
    @no_debug_info_test
    @skipIf(remote=False)
    @expectedFailureAll(hostoslist=["windows"], triple='.*-android')
    def test_platform_process_connect(self):
        self.build()
        self.init_llgs_test(False)

        working_dir = lldb.remote_platform.GetWorkingDirectory()
        src = lldb.SBFileSpec(self.getBuildArtifact("a.out"))
        dest = lldb.SBFileSpec(os.path.join(working_dir, "a.out"))
        err = lldb.remote_platform.Put(src, dest)
        if err.Fail():
            raise RuntimeError(
                "Unable copy '%s' to '%s'.\n>>> %s" %
                (f, wd, err.GetCString()))

        m = re.search("^(.*)://([^:/]*)", configuration.lldb_platform_url)
        protocol = m.group(1)
        hostname = m.group(2)
        unix_protocol = protocol.startswith("unix-")
        if unix_protocol:
            p = re.search("^(.*)-connect", protocol)
            path = lldbutil.join_remote_paths(configuration.lldb_platform_working_dir,
                    self.getBuildDirBasename(), "platform-%d.sock" % int(time.time()))
            listen_url = "%s://%s" % (p.group(1), path)
        else:
            listen_url = "*:0"

        port_file = "%s/port" % working_dir
        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--socket-file",
            port_file,
            "--",
            "%s/a.out" %
            working_dir,
            "foo"]
        self.spawnSubprocess(
            self.debug_monitor_exe,
            commandline_args,
            install_remote=False)
        self.addTearDownHook(self.cleanupSubprocesses)

        socket_id = lldbutil.wait_for_file_on_target(self, port_file)

        new_debugger = lldb.SBDebugger.Create()
        new_debugger.SetAsync(False)

        def del_debugger(new_debugger=new_debugger):
            del new_debugger
        self.addTearDownHook(del_debugger)

        new_platform = lldb.SBPlatform(lldb.remote_platform.GetName())
        new_debugger.SetSelectedPlatform(new_platform)
        new_interpreter = new_debugger.GetCommandInterpreter()

        if unix_protocol:
            connect_url = "%s://%s%s" % (protocol, hostname, socket_id)
        else:
            connect_url = "%s://%s:%s" % (protocol, hostname, socket_id)

        command = "platform connect %s" % (connect_url)
        result = lldb.SBCommandReturnObject()
        new_interpreter.HandleCommand(command, result)
        self.assertTrue(
            result.Succeeded(),
            "platform process connect failed: %s" %
            result.GetOutput())

        target = new_debugger.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)

        breakpoint = target.BreakpointCreateByName("main")
        process.Continue()

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunction().GetName(), "main")
        self.assertEqual(frame.FindVariable("argc").GetValueAsSigned(), 2)
        process.Continue()
