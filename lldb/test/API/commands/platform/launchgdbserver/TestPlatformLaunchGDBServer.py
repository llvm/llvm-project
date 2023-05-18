""" Check that errors while handling qLaunchGDBServer are reported to the user.
    Though this isn't a platform command in itself, the best way to test it is
    from Python because we can juggle multiple processes more easily.
"""

import os
import socket
import shutil
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPlatformProcessLaunchGDBServer(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    # Windows doesn't use lldb-server and on Darwin we may be using debugserver.
    @skipUnlessPlatform(["linux"])
    @add_test_categories(["lldb-server"])
    def test_platform_process_launch_gdb_server(self):
        self.build()

        hostname = socket.getaddrinfo("localhost", 0, proto=socket.IPPROTO_TCP)[0][4][0]
        listen_url = "[%s]:0" % hostname

        port_file = self.getBuildArtifact("port")
        commandline_args = [
            "platform",
            "--listen",
            listen_url,
            "--socket-file",
            port_file,
            "--",
            self.getBuildArtifact("a.out"),
            "foo",
        ]

        # Run lldb-server from a new location.
        new_lldb_server = self.getBuildArtifact("lldb-server")
        shutil.copy(lldbgdbserverutils.get_lldb_server_exe(), new_lldb_server)

        self.spawnSubprocess(new_lldb_server, commandline_args)
        socket_id = lldbutil.wait_for_file_on_target(self, port_file)

        # Remove our new lldb-server so that when it tries to invoke itself as a
        # gdbserver, it fails.
        os.remove(new_lldb_server)

        new_platform = lldb.SBPlatform("remote-" + self.getPlatform())
        self.dbg.SetSelectedPlatform(new_platform)

        connect_url = "connect://[%s]:%s" % (hostname, socket_id)
        self.runCmd("platform connect %s" % connect_url)

        self.runCmd("target create {}".format(self.getBuildArtifact("a.out")))
        self.expect("run", substrs=["unable to launch a GDB server on"], error=True)
