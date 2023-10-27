"""
Test trimming long progress report in tiny terminal windows
"""

import os
import pexpect
import tempfile
import re

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class TestTrimmedProgressReporting(PExpectTest):
    def do_test(self, term_width, pattern_list):
        self.build()
        # Start with a small window
        self.launch(use_colors=True)
        self.expect("set set show-progress true")
        self.expect(
            "set show show-progress", substrs=["show-progress (boolean) = true"]
        )
        self.expect("set set term-width " + str(term_width))
        self.expect(
            "set show term-width", substrs=["term-width (int) = " + str(term_width)]
        )

        self.child.send("file " + self.getBuildArtifact("a.out") + "\n")
        self.child.expect(pattern_list)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipUnlessDarwin
    @skipIfEditlineSupportMissing
    def test_trimmed_progress_message(self):
        self.do_test(
            19, ["Locating externa...", "Loading Apple DW...", "Parsing symbol t..."]
        )

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipUnlessDarwin
    @skipIfEditlineSupportMissing
    def test_long_progress_message(self):
        self.do_test(
            80,
            [
                "Locating external symbol file for a.out...",
                "Loading Apple DWARF index for a.out...",
                "Parsing symbol table for dyld...",
            ],
        )
