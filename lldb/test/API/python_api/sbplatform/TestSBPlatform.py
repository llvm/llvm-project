"""Test the SBPlatform APIs."""

import os
import stat
from pathlib import Path
import lldbgdbserverutils

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWasm  # no remote platform file/process APIs
class SBPlatformAPICase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote  # Remote environment not supported.
    def test_run(self):
        self.build()
        plat = lldb.SBPlatform.GetHostPlatform()

        os.environ["MY_TEST_ENV_VAR"] = "SBPlatformAPICase.test_run"

        def cleanup():
            del os.environ["MY_TEST_ENV_VAR"]

        self.addTearDownHook(cleanup)
        cmd = lldb.SBPlatformShellCommand(self.getBuildArtifact("a.out"))
        self.assertSuccess(plat.Run(cmd))
        self.assertIn("MY_TEST_ENV_VAR=SBPlatformAPICase.test_run", cmd.GetOutput())

    def test_SetSDKRoot(self):
        plat = lldb.SBPlatform("remote-linux")  # arbitrary choice
        self.assertTrue(plat)
        plat.SetSDKRoot(self.getBuildDir())
        self.dbg.SetSelectedPlatform(plat)
        self.expect("platform status", substrs=["Sysroot:", self.getBuildDir()])

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

    def do_copy_test(self, platform):
        source_name = "source-file"
        source_path = self.getBuildArtifact(source_name)
        Path(source_path).touch()

        remote_path = lldbutil.append_to_process_working_directory(self, source_name)
        put_error = platform.Put(
            lldb.SBFileSpec(source_path, True), lldb.SBFileSpec(remote_path, False)
        )
        self.assertSuccess(put_error)

        destination = self.getBuildArtifact("destination-file")
        # Make the file read only.
        Path(destination).touch(mode=stat.S_IREAD, exist_ok=True)

        def remove_destination_file():
            os.chmod(destination, stat.S_IWRITE)
            os.remove(destination)

        self.addTearDownHook(remove_destination_file)

        get_error = platform.Get(
            lldb.SBFileSpec(remote_path, False), lldb.SBFileSpec(destination, True)
        )
        # Ideally we would check that we failed due to permissions, but as this
        # runs on many platforms, we do not know the exact form of the error message.
        self.assertFailure(get_error)

    @skipIfDarwin  # lldb-server not found correctly
    @add_test_categories(["lldb-server"])
    # If we're already remote, no need to spawn a new remote platform.
    @skipIfRemote
    def test_get_reports_write_failure_remote(self):
        if lldb.remote_platform:
            platform = lldb.remote_platform
        else:
            platform = lldbutil.connect_to_new_remote_platform(
                self, lldbgdbserverutils.get_lldb_server_exe()
            )

        self.do_copy_test(platform)

    @skipIfDarwin  # lldb-server not found correctly
    @add_test_categories(["lldb-server"])
    def test_get_reports_write_failure(self):
        self.do_copy_test(self.dbg.GetSelectedPlatform())
