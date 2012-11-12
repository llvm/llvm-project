"""
Test lldb 'process connect' command.
"""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class ConnectRemoteTestCase(TestBase):

    mydir = os.path.join("functionalities", "connect_remote")

    def test_connect_remote(self):
        """Test "process connect connect:://localhost:12345"."""

        # First, we'll start a fake debugserver (a simple echo server).
        fakeserver = pexpect.spawn('./EchoServer.py')

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            fakeserver.logfile_read = sys.stdout

        # Schedule the fake debugserver to be shutting down during teardown.
        def shutdown_fakeserver():
            fakeserver.close()
        self.addTearDownHook(shutdown_fakeserver)

        # Wait until we receive the server ready message before continuing.
        fakeserver.expect_exact('Listening on localhost:12345')

        # Connect to the fake server....
        self.runCmd("process connect connect://localhost:12345")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
