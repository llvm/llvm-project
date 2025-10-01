import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MissingDllTestCase(TestBase):
    @skipUnlessWindows
    def test(self):
        """
        Test that lldb reports the application's exit code (STATUS_DLL_NOT_FOUND),
        rather than trying to treat it as a Win32 error number.
        """

        self.build()
        exe = self.getBuildArtifact("a.out")
        dll = self.getBuildArtifact("dummy_dll.dll")
        self.assertTrue(remove_file(dll))
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())

        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertFailure(error, "Process prematurely exited with 0xc0000135")
