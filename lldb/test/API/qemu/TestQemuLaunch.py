from __future__ import print_function
import lldb
import unittest
import os
import json
import stat
import sys
from textwrap import dedent
import lldbsuite.test.lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *


@skipIfRemote
@skipIfWindows
class TestQemuLaunch(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def set_emulator_setting(self, name, value):
        self.runCmd("settings set platform.plugin.qemu-user.%s %s" %
                (name, value))

    def setUp(self):
        super().setUp()
        emulator = self.getBuildArtifact("qemu.py")
        with os.fdopen(os.open(emulator, os.O_WRONLY|os.O_CREAT, stat.S_IRWXU),
                "w") as e:

            e.write(dedent("""\
                    #! {exec!s}

                    import runpy
                    import sys

                    sys.path = {path!r}
                    runpy.run_path({source!r}, run_name='__main__')
                    """).format(exec=sys.executable, path=sys.path,
                        source=self.getSourcePath("qemu.py")))

        self.set_emulator_setting("architecture", self.getArchitecture())
        self.set_emulator_setting("emulator-path", emulator)

    def test_basic_launch(self):
        self.build()
        exe = self.getBuildArtifact()

        # Create a target using our platform
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertSuccess(error)
        self.assertEqual(target.GetPlatform().GetName(), "qemu-user")

        # "Launch" the process. Our fake qemu implementation will pretend it
        # immediately exited.
        process = target.LaunchSimple(
                ["dump:" + self.getBuildArtifact("state.log")], None, None)
        self.assertIsNotNone(process)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0x47)

        # Verify the qemu invocation parameters.
        with open(self.getBuildArtifact("state.log")) as s:
            state = json.load(s)
        self.assertEqual(state["program"], self.getBuildArtifact())
        self.assertEqual(state["args"],
                ["dump:" + self.getBuildArtifact("state.log")])

    def test_stdio_pty(self):
        self.build()
        exe = self.getBuildArtifact()

        # Create a target using our platform
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertSuccess(error)

        info = lldb.SBLaunchInfo([
            "stdin:stdin",
            "stdout:STDOUT CONTENT\n",
            "stderr:STDERR CONTENT\n",
            "dump:" + self.getBuildArtifact("state.log"),
            ])

        listener = lldb.SBListener("test_stdio")
        info.SetListener(listener)

        self.dbg.SetAsync(True)
        process = target.Launch(info, error)
        self.assertSuccess(error)
        lldbutil.expect_state_changes(self, listener, process,
                [lldb.eStateRunning])

        process.PutSTDIN("STDIN CONTENT\n")

        lldbutil.expect_state_changes(self, listener, process,
                [lldb.eStateExited])

        # Echoed stdin, stdout and stderr. With a pty we cannot split standard
        # output and error.
        self.assertEqual(process.GetSTDOUT(1000),
                "STDIN CONTENT\r\nSTDOUT CONTENT\r\nSTDERR CONTENT\r\n")
        with open(self.getBuildArtifact("state.log")) as s:
            state = json.load(s)
        self.assertEqual(state["stdin"], "STDIN CONTENT\n")

    def test_stdio_redirect(self):
        self.build()
        exe = self.getBuildArtifact()

        # Create a target using our platform
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertSuccess(error)

        info = lldb.SBLaunchInfo([
            "stdin:stdin",
            "stdout:STDOUT CONTENT",
            "stderr:STDERR CONTENT",
            "dump:" + self.getBuildArtifact("state.log"),
            ])

        info.AddOpenFileAction(0, self.getBuildArtifact("stdin.txt"),
                True, False)
        info.AddOpenFileAction(1, self.getBuildArtifact("stdout.txt"),
                False, True)
        info.AddOpenFileAction(2, self.getBuildArtifact("stderr.txt"),
                False, True)

        with open(self.getBuildArtifact("stdin.txt"), "w") as f:
            f.write("STDIN CONTENT")

        process = target.Launch(info, error)
        self.assertSuccess(error)
        self.assertEqual(process.GetState(), lldb.eStateExited)

        with open(self.getBuildArtifact("stdout.txt")) as f:
            self.assertEqual(f.read(), "STDOUT CONTENT")
        with open(self.getBuildArtifact("stderr.txt")) as f:
            self.assertEqual(f.read(), "STDERR CONTENT")
        with open(self.getBuildArtifact("state.log")) as s:
            state = json.load(s)
        self.assertEqual(state["stdin"], "STDIN CONTENT")

    def test_bad_emulator_path(self):
        self.set_emulator_setting("emulator-path",
                self.getBuildArtifact("nonexistent.file"))

        self.build()
        exe = self.getBuildArtifact()

        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertEqual(target.GetPlatform().GetName(), "qemu-user")
        self.assertSuccess(error)
        info = lldb.SBLaunchInfo([])
        target.Launch(info, error)
        self.assertTrue(error.Fail())
        self.assertIn("doesn't exist", error.GetCString())
