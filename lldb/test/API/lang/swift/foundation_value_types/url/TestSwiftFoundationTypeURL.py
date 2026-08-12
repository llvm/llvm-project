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
    @requireNotEmbeddedSwift
    @expectedFailureAll(archs=["arm64_32"], bugnumber="<rdar://problem/58065423>")
    @skipUnlessFoundationEssentials
    @skipIfLinux  # https://github.com/swiftlang/llvm-project/issues/13465
    @swiftTest
    def test_swift_url_formatters(self):
        """Test URL summary strings."""
        self.build()

        foundation = "Foundation" if sys.platform == "darwin" else "FoundationEssentials"

        # Foundation's URL stores its NSURL in a Synchronization.Mutex
        # whose reflection metadata is missing if the stdlib is
        # compiled with a deployment target < macOS 27.0, which is the
        # case when building locally. Debug against the system
        # Foundation / Swift stdlib (whose deployment target matches
        # the OS) by dropping the library-path variables the test
        # harness injects into the inferior environment (via
        # --inferior-env) before launching.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        for var in [
            "DYLD_LIBRARY_PATH",
            "LD_LIBRARY_PATH",
            "SIMCTL_CHILD_DYLD_LIBRARY_PATH",
        ]:
            self.runCmd(f"settings remove target.env-vars {var}", check=False)
        bkpt = target.BreakpointCreateBySourceRegex(
            "break here", lldb.SBFileSpec("main.swift")
        )
        self.assertGreater(bkpt.GetNumLocations(), 0)
        lldbutil.run_to_breakpoint_do_run(self, target, bkpt)

        self.expect(
            "frame var url",
            substrs=[
                f"({foundation}.URL?)",
                "url",
                'https://www.example.com/path?query#fragment',
            ],
        )
        self.expect(
            "expression -d run -- url",
            substrs=[
                f"({foundation}.URL?)",
                "https://www.example.com/path?query#fragment",
            ],
        )

        self.expect(
            "frame var relativeURL",
            substrs=[
                f"({foundation}.URL?)",
                "relativeURL",
                "relative",
                "--",
                "https://www.example.com/",
            ],
        )
        self.expect(
            "expression -d run -- relativeURL",
            substrs=[
                f"({foundation}.URL?)",
                "relative",
                "--",
                "https://www.example.com/",
            ],
        )

        self.expect(
            "frame var g_url",
            substrs=[f"({foundation}.URL)", "g_url", "http://www.apple.com"],
        )
        self.expect(
            "expression -d run -- g_url",
            substrs=[f"({foundation}.URL)", "http://www.apple.com"],
        )
