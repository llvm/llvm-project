"""
Test the error message if the SDK the program was built against doesn't exist.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestMissingSDK(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @swiftTest
    @skipIf(oslist=['windows'])
    @skipIfDarwinEmbedded # swift crash inspecting swift stdlib with little other swift loaded <rdar://problem/55079456> 
    def testMissingSDK(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        self.expect("p message", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs = ["Hello"])

