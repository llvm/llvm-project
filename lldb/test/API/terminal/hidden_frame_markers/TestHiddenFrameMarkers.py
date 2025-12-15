"""
Test that hidden frames are delimited with markers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class HiddenFrameMarkerTest(TestBase):
    @unicodeTest
    def test_hidden_frame_markers(self):
        """Test that hidden frame markers are rendered in backtraces"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("bt", substrs=["﹍frame #1:", "﹉frame #7:", "　frame #8:", "　frame #9:"])
