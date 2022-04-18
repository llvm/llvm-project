"""Test the SBPlatform APIs."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class SBPlatformAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote # Remote environment not supported.
    def test_run(self):
        self.build()
        plat = lldb.SBPlatform.GetHostPlatform()

        os.environ["MY_TEST_ENV_VAR"]="SBPlatformAPICase.test_run"
        def cleanup():
            del os.environ["MY_TEST_ENV_VAR"]
        self.addTearDownHook(cleanup)
        cmd = lldb.SBPlatformShellCommand(self.getBuildArtifact("a.out"))
        self.assertSuccess(plat.Run(cmd))
        self.assertIn("MY_TEST_ENV_VAR=SBPlatformAPICase.test_run", cmd.GetOutput())

    def test_SetSDKRoot(self):
        plat = lldb.SBPlatform("remote-linux") # arbitrary choice
        self.assertTrue(plat)
        plat.SetSDKRoot(self.getBuildDir())
        self.dbg.SetSelectedPlatform(plat)
        self.expect("platform status",
                substrs=["Sysroot:", self.getBuildDir()])

    def test_SetCurrentPlatform_floating(self):
        # floating platforms cannot be referenced by name until they are
        # associated with a debugger
        floating_platform = lldb.SBPlatform("remote-netbsd")
        floating_platform.SetWorkingDirectory(self.getBuildDir())
        self.assertSuccess(self.dbg.SetCurrentPlatform("remote-netbsd"))
        dbg_platform = self.dbg.GetSelectedPlatform()
        self.assertEqual(dbg_platform.GetName(), "remote-netbsd")
        self.assertIsNone(dbg_platform.GetWorkingDirectory())

    def test_SetCurrentPlatform_associated(self):
        # associated platforms are found by name-based lookup
        floating_platform = lldb.SBPlatform("remote-netbsd")
        floating_platform.SetWorkingDirectory(self.getBuildDir())
        orig_platform = self.dbg.GetSelectedPlatform()

        self.dbg.SetSelectedPlatform(floating_platform)
        self.dbg.SetSelectedPlatform(orig_platform)
        self.assertSuccess(self.dbg.SetCurrentPlatform("remote-netbsd"))
        dbg_platform = self.dbg.GetSelectedPlatform()
        self.assertEqual(dbg_platform.GetName(), "remote-netbsd")
        self.assertEqual(dbg_platform.GetWorkingDirectory(), self.getBuildDir())
