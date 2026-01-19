"""
Test that hidden frames are delimited with markers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class HiddenFrameMarkerTest(TestBase):
    @unicode_test
    def test_hidden_frame_markers(self):
        """Test that hidden frame markers are rendered in backtraces"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "bt",
            substrs=[
                "   * frame #0:",
                "  ﹍ frame #1:",
                "  ﹉ frame #7:",
                "     frame #8:",
                "     frame #9:",
            ],
        )

        self.runCmd("f 1")
        self.expect(
            "bt",
            substrs=[
                "     frame #0:",
                "   * frame #1:",
                "  ﹉ frame #7:",
                "     frame #8:",
                "     frame #9:",
            ],
        )

        self.runCmd("f 7")
        self.expect(
            "bt",
            substrs=[
                "     frame #0:",
                "  ﹍ frame #1:",
                "   * frame #7:",
                "     frame #8:",
                "     frame #9:",
            ],
        )

    def test_hidden_frame_markers(self):
        """
        Test that hidden frame markers are not rendered in backtraces when
        mark-hidden-frames is set to false
        """
        self.build()
        self.runCmd("settings set mark-hidden-frames 0")
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "bt",
            substrs=[
                "  * frame #0:",
                "    frame #1:",
                "    frame #7:",
                "    frame #8:",
                "    frame #9:",
            ],
        )

        self.runCmd("f 1")
        self.expect(
            "bt",
            substrs=[
                "    frame #0:",
                "  * frame #1:",
                "    frame #7:",
                "    frame #8:",
                "    frame #9:",
            ],
        )

        self.runCmd("f 7")
        self.expect(
            "bt",
            substrs=[
                "    frame #0:",
                "    frame #1:",
                "  * frame #7:",
                "    frame #8:",
                "    frame #9:",
            ],
        )
