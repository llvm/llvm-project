"""
Test that an OS plugin in a dSYM sees the right process state
when run from a dSYM on attach
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbgdbserverutils import get_debugserver_exe

import os
import lldb
import time
import socket
import shutil


class TestOSPluginIndSYM(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # The port used by debugserver.
    PORT = 54638

    # The number of attempts.
    ATTEMPTS = 10

    # Time given to the binary to launch and to debugserver to attach to it for
    # every attempt. We'll wait a maximum of 10 times 2 seconds while the
    # inferior will wait 10 times 10 seconds.
    TIMEOUT = 2

    def no_debugserver(self):
        if get_debugserver_exe() is None:
            return "no debugserver"
        return None

    def port_not_available(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if s.connect_ex(("127.0.0.1", self.PORT)) == 0:
            return "{} not available".format(self.PORT)
        return None

    @skipUnlessDarwin
    def test_python_os_plugin(self):
        self.do_test_python_os_plugin(False)

    @skipTestIfFn(no_debugserver)
    @skipTestIfFn(port_not_available)
    def test_python_os_plugin_remote(self):
        self.do_test_python_os_plugin(True)

    def do_test_python_os_plugin(self, remote):
        """Test that the environment for os plugins in dSYM's is correct"""
        executable = self.build_dsym("my_binary")

        # Make sure we're set up to load the symbol file's python
        self.runCmd("settings set target.load-script-from-symbol-file true")

        target = self.dbg.CreateTarget(None)

        error = lldb.SBError()

        # Now run the process, and then attach.  When the attach
        # succeeds, make sure that we were in the right state when
        # the OS plugins were run.
        if not remote:
            popen = self.spawnSubprocess(executable, [])

            process = target.AttachToProcessWithID(lldb.SBListener(), popen.pid, error)
            self.assertSuccess(error, "Attach succeeded")
        else:
            self.setup_remote_platform(executable)
            process = target.process
            self.assertTrue(process.IsValid(), "Got a valid process from debugserver")

        # We should have figured out the target from the result of the attach:
        self.assertTrue(target.IsValid, "Got a valid target")

        # Make sure that we got the right plugin:
        self.expect(
            "settings show target.process.python-os-plugin-path",
            substrs=["operating_system.py"],
        )

        for thread in process.threads:
            stack_depth = thread.num_frames
            reg_threads = thread.frames[0].reg

        # OKAY, that realized the threads, now see if the creation
        # state was correct.  The way we use the OS plugin, it doesn't need
        # to create a thread, and doesn't have to call get_register_info,
        # so we don't expect those to get called.
        self.expect(
            "test_report_command",
            substrs=[
                "in_init=1",
                "in_get_thread_info=1",
                "in_create_thread=2",
                "in_get_register_info=2",
                "in_get_register_data=1",
            ],
        )

    def build_dsym(self, name):
        self.build(debug_info="dsym", dictionary={"EXE": name})
        executable = self.getBuildArtifact(name)
        dsym_path = self.getBuildArtifact(name + ".dSYM")
        python_dir_path = dsym_path
        python_dir_path = os.path.join(dsym_path, "Contents", "Resources", "Python")
        if not os.path.exists(python_dir_path):
            os.mkdir(python_dir_path)
        python_file_name = name + ".py"

        os_plugin_dir = os.path.join(python_dir_path, "OS_Plugin")
        if not os.path.exists(os_plugin_dir):
            os.mkdir(os_plugin_dir)

        plugin_dest_path = os.path.join(os_plugin_dir, "operating_system.py")
        plugin_origin_path = os.path.join(self.getSourceDir(), "operating_system.py")
        shutil.copy(plugin_origin_path, plugin_dest_path)

        module_dest_path = os.path.join(python_dir_path, python_file_name)
        with open(module_dest_path, "w") as f:
            f.write("def __lldb_init_module(debugger, unused):\n")
            f.write(
                f"    debugger.HandleCommand(\"settings set target.process.python-os-plugin-path '{plugin_dest_path}'\")\n"
            )
            f.close()

        return executable

    def setup_remote_platform(self, exe):
        # Get debugserver to start up our process for us, and then we
        # can use `process connect` to attach to it.
        debugserver = get_debugserver_exe()
        debugserver_args = ["localhost:{}".format(self.PORT), exe]
        self.spawnSubprocess(debugserver, debugserver_args)

        # Select the platform.
        self.runCmd("platform select remote-gdb-server")

        # Connect to debugserver
        interpreter = self.dbg.GetCommandInterpreter()
        connected = False
        for i in range(self.ATTEMPTS):
            result = lldb.SBCommandReturnObject()
            interpreter.HandleCommand(f"gdb-remote localhost:{self.PORT}", result)
            connected = result.Succeeded()
            if connected:
                break
            time.sleep(self.TIMEOUT)

        self.assertTrue(connected, "could not connect to debugserver")
