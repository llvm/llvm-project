"""
Test that if you exec lldb with the stdio file handles
closed, it is able to exit without hanging.
"""


import lldb
import os
import sys
import socket

if os.name != "nt":
    import fcntl

import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestDriverWithClosedSTDIO(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    # Windows doesn't have the fcntl module, so we can't run this
    # test there.
    @skipIf(hostoslist=["windows"])
    def test_run_lldb_and_wait(self):
        """This test forks, closes the stdio channels and exec's lldb.
        Then it waits for it to exit and asserts it did that successfully"""
        pid = os.fork()
        if pid == 0:
            fcntl.fcntl(sys.stdin, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
            fcntl.fcntl(sys.stdout, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
            fcntl.fcntl(sys.stderr, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
            lldb = lldbtest_config.lldbExec
            print(f"About to run: {lldb}")
            os.execlp(
                lldb,
                lldb,
                "-x",
                "-o",
                "script print(lldb.debugger.GetNumTargets())",
                "--batch",
            )
        else:
            if pid == -1:
                print("Couldn't fork a process.")
                return
            ret_pid, status = os.waitpid(pid, 0)
            # We're really just checking that lldb doesn't stall.
            # At the time this test was written, if you close stdin
            # in an asserts build, lldb aborts.  So handle both
            # of those cases.  The failure will just be that the
            # waitpid doesn't return, and the test times out.
            self.assertFalse(os.WIFSTOPPED(status), "We either exited or crashed.")
