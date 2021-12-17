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
        self.runCmd("settings set -- platform.plugin.qemu-user.%s %s" %
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

    def _create_target(self):
        self.build()
        exe = self.getBuildArtifact()

        # Create a target using our platform
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, '', 'qemu-user', False, error)
        self.assertSuccess(error)
        self.assertEqual(target.GetPlatform().GetName(), "qemu-user")
        return target

    def _run_and_get_state(self, target=None, info=None):
        if target is None:
            target = self._create_target()

        if info is None:
            info = target.GetLaunchInfo()

        # "Launch" the process. Our fake qemu implementation will pretend it
        # immediately exited.
        info.SetArguments(["dump:" + self.getBuildArtifact("state.log")], True)
        error = lldb.SBError()
        process = target.Launch(info, error)
        self.assertSuccess(error)
        self.assertIsNotNone(process)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0x47)

        # Verify the qemu invocation parameters.
        with open(self.getBuildArtifact("state.log")) as s:
            return json.load(s)

    def test_basic_launch(self):
        state = self._run_and_get_state()

        self.assertEqual(state["program"], self.getBuildArtifact())
        self.assertEqual(state["args"],
                ["dump:" + self.getBuildArtifact("state.log")])

    def test_stdio_pty(self):
        target = self._create_target()

        info = target.GetLaunchInfo()
        info.SetArguments([
            "stdin:stdin",
            "stdout:STDOUT CONTENT\n",
            "stderr:STDERR CONTENT\n",
            "dump:" + self.getBuildArtifact("state.log"),
            ], False)

        listener = lldb.SBListener("test_stdio")
        info.SetListener(listener)

        self.dbg.SetAsync(True)
        error = lldb.SBError()
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

        target = self._create_target()
        info = lldb.SBLaunchInfo([])
        error = lldb.SBError()
        target.Launch(info, error)
        self.assertTrue(error.Fail())
        self.assertIn("doesn't exist", error.GetCString())

    def test_extra_args(self):
        self.set_emulator_setting("emulator-args", "-fake-arg fake-value")
        state = self._run_and_get_state()

        self.assertEqual(state["fake-arg"], "fake-value")

    def test_env_vars(self):
        # First clear any global environment to have a clean slate for this test
        self.runCmd("settings clear target.env-vars")
        self.runCmd("settings clear target.unset-env-vars")

        def var(i):
            return "LLDB_TEST_QEMU_VAR%d" % i

        # Set some variables in the host environment.
        for i in range(4):
            os.environ[var(i)]="from host"
        def cleanup():
            for i in range(4):
                del os.environ[var(i)]
        self.addTearDownHook(cleanup)

        # Set some emulator-only variables.
        self.set_emulator_setting("emulator-env-vars",
                "%s='emulator only'"%var(4))

        # And through the platform setting.
        self.set_emulator_setting("target-env-vars",
                "%s='from platform' %s='from platform'" % (var(1), var(2)))

        target = self._create_target()
        info = target.GetLaunchInfo()
        env = info.GetEnvironment()

        # Platform settings should trump host values. Emulator-only variables
        # should not be visible.
        self.assertEqual(env.Get(var(0)), "from host")
        self.assertEqual(env.Get(var(1)), "from platform")
        self.assertEqual(env.Get(var(2)), "from platform")
        self.assertEqual(env.Get(var(3)), "from host")
        self.assertIsNone(env.Get(var(4)))

        # Finally, make some launch_info specific changes.
        env.Set(var(2), "from target", True)
        env.Unset(var(3))
        info.SetEnvironment(env, False)

        # Now check everything. Launch info changes should trump everything, but
        # only for the target environment -- the emulator should still get the
        # host values.
        state = self._run_and_get_state(target, info)
        for i in range(4):
            self.assertEqual(state["environ"][var(i)], "from host")
        self.assertEqual(state["environ"][var(4)], "emulator only")
        self.assertEqual(state["environ"]["QEMU_SET_ENV"],
                "%s=from platform,%s=from target" % (var(1), var(2)))
        self.assertEqual(state["environ"]["QEMU_UNSET_ENV"],
                "%s,%s,QEMU_SET_ENV,QEMU_UNSET_ENV" % (var(3), var(4)))
