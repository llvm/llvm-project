"""Test the SBInitializationOptions API."""

import lldb
from lldbsuite.test.lldbtest import *


class SBInitializationOptionsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_initialize(self):
        options = lldb.SBInitializationOptions()
        self.assertSuccess(lldb.SBDebugger.Initialize(options))
