# -*- coding: utf-8 -*-
"""
Test unicode handling in LLDB.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestCase(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @expectedFailureAll()
    @skipIfAsan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    def test_unicode_input(self):
        self.launch()

        # Send some unicode input to LLDB.
        # We should get back that this is an invalid command with our character as UTF-8.
        self.expect(u'\u1234', substrs=[u"error: '\u1234' is not a valid command.".encode('utf-8')])

        self.quit()
