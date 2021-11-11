from __future__ import print_function
import lldb
import unittest
import os
import json
import stat
import sys
from textwrap import dedent
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

        # Create a target using out platform
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertSuccess(error)
        self.assertEqual(target.GetPlatform().GetName(), "qemu-user")

        # "Launch" the process. Our fake qemu implementation will pretend it
        # immediately exited.
        process = target.LaunchSimple(
                [self.getBuildArtifact("state.log"), "arg2", "arg3"], None, None)
        self.assertIsNotNone(process)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0x47)

        # Verify the qemu invocation parameters.
        with open(self.getBuildArtifact("state.log")) as s:
            state = json.load(s)
        self.assertEqual(state["program"], self.getBuildArtifact())
        self.assertEqual(state["rest"], ["arg2", "arg3"])

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
