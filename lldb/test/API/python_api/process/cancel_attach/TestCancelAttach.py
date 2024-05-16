"""
Test using SendAsyncInterrupt to interrupt an "attach wait"
"""

import lldb
import sys
import time
import threading
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil


class AttachCancelTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_scripted_implementation(self):
        """Test that cancelling a stuck "attach waitfor" works."""
        # First make an empty target for the attach:
        target = self.dbg.CreateTarget(None)

        # We need to cancel this, so we need to do the attach
        # on a separate thread:
        class AttachThread(threading.Thread):
            def __init__(self, target, error):
                # Make this a daemon thread so if we don't manage to interrupt,
                # Python will keep this thread from hanging the test.
                threading.Thread.__init__(self, daemon=True)
                self.target = target
                self.error = error

            def run(self):
                self.target.AttachToProcessWithName(
                    lldb.SBListener(), "LLDB-No-Such-Process", True, self.error
                )

        error = lldb.SBError()
        thread = AttachThread(target, error)
        thread.start()

        # Now wait till the attach on the child thread has made a process
        # for the attach attempt:
        while not target.process.IsValid():
            time.sleep(1)
        # I don't have a positive signal for "we started the attach attempt"
        # so the best I can do is sleep a bit more here to give that a chance
        # to start:
        time.sleep(1)

        # Now send the attach interrupt:
        target.process.SendAsyncInterrupt()
        # We don't want to stall if we can't interrupt, so join with a timeout:
        thread.join(60)
        if thread.is_alive():
            self.fail("The attach thread is alive after timeout interval")

        # Now check the error, should say the attach was interrupted:
        self.assertTrue(error.Fail(), "We succeeded in not attaching")
