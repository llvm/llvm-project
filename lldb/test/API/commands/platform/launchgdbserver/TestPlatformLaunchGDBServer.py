import os
import shutil
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWasm  # cannot launch a gdbserver
class TestPlatformProcessLaunchGDBServer(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def _launch_and_connect(self, exe):
        new_platform = lldbutil.connect_to_new_remote_platform(self, exe)
        wd = self.getBuildArtifact("wd")
        os.mkdir(wd)
        new_platform.SetWorkingDirectory(wd)

    @skipIfRemote
    # Windows cannot delete the executable while it is running.
    # On Darwin we may be using debugserver.
    @skipUnlessPlatform(["linux"])
    @add_test_categories(["lldb-server"])
    def test_launch_error(self):
        """
        Check that errors while handling qLaunchGDBServer are reported to the
        user.  Though this isn't a platform command in itself, the best way to
        test it is from Python because we can juggle multiple processes more
        easily.
        """

        self.build()

        # Run lldb-server from a new location.
        new_lldb_server = self.getBuildArtifact("lldb-server")
        shutil.copy(lldbgdbserverutils.get_lldb_server_exe(), new_lldb_server)
        self._launch_and_connect(new_lldb_server)

        # Now, remove our new lldb-server so that when it tries to invoke itself as a
        # gdbserver, it fails.
        os.remove(new_lldb_server)

        self.runCmd("target create {}".format(self.getBuildArtifact("a.out")))
        self.expect("run", substrs=["unable to launch a GDB server on"], error=True)

    @skipIfRemote
    @skipIfDarwin  # Uses debugserver for debugging
    @skipIfWindows
    @add_test_categories(["lldb-server"])
    def test_launch_with_unusual_process_name(self):
        """
        Test that lldb-server can launch a debug session when running under an
        unusual name (or under a symlink which resolves to an unusal name).
        """

        self.build()

        # Run lldb-server from a new location.
        new_lldb_server = self.getBuildArtifact("obfuscated-server")
        shutil.copy(lldbgdbserverutils.get_lldb_server_exe(), new_lldb_server)
        self._launch_and_connect(new_lldb_server)

        self.runCmd("target create {}".format(self.getBuildArtifact("a.out")))
        self.expect("run", substrs=["exited with status = 0"])
