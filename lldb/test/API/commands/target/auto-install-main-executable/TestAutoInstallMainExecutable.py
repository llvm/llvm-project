"""
Test target commands: target.auto-install-main-executable.
"""

import time
import lldbgdbserverutils

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWasm  # no remote platform to auto-install onto
class TestAutoInstallMainExecutable(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipIfRemote
    @skipIfWindows  # This test is flaky on Windows
    def test_target_auto_install_main_executable(self):
        if lldbgdbserverutils.get_lldb_server_exe() is None:
            self.skipTest("lldb-server not found")
        self.build()

        new_platform = lldbutil.connect_to_new_remote_platform(
            self, lldbgdbserverutils.get_lldb_server_exe()
        )

        wd = self.getBuildArtifact("wd")
        os.mkdir(wd)
        new_platform.SetWorkingDirectory(wd)

        # Manually install the modified binary.
        src_device = lldb.SBFileSpec(self.getBuildArtifact("a.device.out"))
        dest = lldb.SBFileSpec(os.path.join(wd, "a.out"))
        self.assertSuccess(new_platform.Put(src_device, dest))

        # Test the default setting.
        self.expect(
            "settings show target.auto-install-main-executable",
            substrs=["target.auto-install-main-executable (boolean) = true"],
            msg="Default settings for target.auto-install-main-executable failed.",
        )

        # Disable the auto install.
        self.runCmd("settings set target.auto-install-main-executable false")
        self.expect(
            "settings show target.auto-install-main-executable",
            substrs=["target.auto-install-main-executable (boolean) = false"],
        )

        # Create the target with the original file.
        self.runCmd(
            "target create --remote-file %s %s "
            % (dest.fullpath, self.getBuildArtifact("a.out"))
        )

        self.expect("process launch", substrs=["exited with status = 74"])
