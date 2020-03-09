"""Tests that a timeout is detected by the testbot."""
from __future__ import print_function

import time

import lldbsuite.test.lldbtest as lldbtest


class TimeoutTestCase(lldbtest.TestBase):
    """Forces test timeout."""
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def test_buildbot_catches_timeout(self):
        """Tests that timeout logic kicks in and is picked up."""
        while True:
            try:
                time.sleep(1)
            except:
                print("ignoring exception during sleep")
