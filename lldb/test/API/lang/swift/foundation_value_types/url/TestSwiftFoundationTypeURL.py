# TestSwiftFoundationTypeURL.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test Foundation.URL summary strings.
"""

import lldb
import sys

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @expectedFailureAll(archs=["arm64_32"], bugnumber="<rdar://problem/58065423>")
    @skipUnlessFoundation
    @swiftTest
    def test_swift_url_formatters(self):
        """Test URL summary strings."""
        self.build()

        foundation = "Foundation" if sys.platform == "darwin" else "FoundationEssentials"

        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame var url",
            startstr=f'({foundation}.URL?) url = "https://www.example.com/path?query#fragment"',
        )
        self.expect(
            "expression -d run -- url",
            startstr=f'({foundation}.URL?) $R0 = "https://www.example.com/path?query#fragment"',
        )

        self.expect(
            "frame var relativeURL",
            startstr=f'({foundation}.URL?) relativeURL = "relative -- https://www.example.com/"',
        )
        self.expect(
            "expression -d run -- relativeURL",
            startstr=f'({foundation}.URL?) $R1 = "relative -- https://www.example.com/"',
        )

        self.expect(
            "frame var g_url",
            startstr=f'({foundation}.URL) g_url = "http://www.apple.com"',
        )
        self.expect(
            "expression -d run -- g_url",
            startstr=f'({foundation}.URL) $R2 = "http://www.apple.com"',
        )
