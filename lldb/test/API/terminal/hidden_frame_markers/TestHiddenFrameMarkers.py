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
            self, "// break here first", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "bt",
            substrs=[
                "   * frame #0:",
                "  ﹉ frame #2:",
                "     frame #3:",
                "     frame #4:",
            ],
        )

        self.runCmd("f 2")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "   * frame #2:",
                "     frame #3:",
                "     frame #4:",
            ],
        )

        self.runCmd("f 3")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "  ﹉ frame #2:",
                "   * frame #3:",
                "     frame #4:",
            ],
        )

    @unicode_test
    def test_nested_hidden_frame_markers(self):
        """Test that nested hidden frame markers are rendered in backtraces"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here after", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "bt",
            substrs=[
                "   * frame #0:",
                "  ﹉ frame #2:",
                "  ﹍ frame #3:",
                "  ﹉ frame #5:",
                "     frame #6:",
            ],
        )

        self.runCmd("f 2")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "   * frame #2:",
                "  ﹍ frame #3:",
                "  ﹉ frame #5:",
                "     frame #6:",
            ],
        )

        self.runCmd("f 3")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "  ﹉ frame #2:",
                "   * frame #3:",
                "  ﹉ frame #5:",
                "     frame #6:",
            ],
        )

        self.runCmd("f 5")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "  ﹉ frame #2:",
                "  ﹍ frame #3:",
                "   * frame #5:",
                "     frame #6:",
            ],
        )

        self.runCmd("f 6")
        self.expect(
            "bt",
            substrs=[
                "  ﹍ frame #0:",
                "  ﹉ frame #2:",
                "  ﹍ frame #3:",
                "  ﹉ frame #5:",
                "   * frame #6:",
            ],
        )

    def test_deactivated_hidden_frame_markers(self):
        """
        Test that hidden frame markers are not rendered in backtraces when
        mark-hidden-frames is set to false
        """
        self.build()
        self.runCmd("settings set mark-hidden-frames 0")
        lldbutil.run_to_source_breakpoint(
            self, "// break here first", lldb.SBFileSpec("main.cpp")
        )
        self.expect(
            "bt",
            substrs=[
                "  * frame #0:",
                "    frame #2:",
                "    frame #3:",
                "    frame #4:",
            ],
        )

        self.runCmd("f 2")
        self.expect(
            "bt",
            substrs=[
                "    frame #0:",
                "  * frame #2:",
                "    frame #3:",
                "    frame #4:",
            ],
        )

        self.runCmd("f 3")
        self.expect(
            "bt",
            substrs=[
                "    frame #0:",
                "    frame #2:",
                "  * frame #3:",
                "    frame #4:",
            ],
        )
