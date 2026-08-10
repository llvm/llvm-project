"""
Test handling of Anonymous Structs, especially that they don't crash lldb.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os
import shutil
import time


class TestFrameVarAnonStruct(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Verify that we don't crash in this case.
        self.expect(
            "target variable 'b.x'",
            error=True,
            substrs=["can't find global variable 'b.x'"],
        )
