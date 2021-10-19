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
        self.assertTrue(plat.Run(cmd).Success())
        self.assertIn("MY_TEST_ENV_VAR=SBPlatformAPICase.test_run", cmd.GetOutput())
