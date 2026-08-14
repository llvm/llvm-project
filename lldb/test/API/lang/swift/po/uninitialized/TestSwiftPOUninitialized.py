# TestSwiftPOUninitialized.py
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
Test that po detects a class reference that has not been assigned yet.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestCase(TestBase):
    @swiftTest
    @requireNotEmbeddedSwift
    def test_po_uninitialized(self):
        """po reports <uninitialized> before the assignment and the instance's
        description after it."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break before assignment", lldb.SBFileSpec("main.swift")
        )

        self.assertIsNone(
            self.frame().FindVariable("object").GetObjectDescription(),
            "po correctly detects uninitialized instances",
        )
        self.expect("po object", substrs=["<uninitialized>"])

        lldbutil.continue_to_source_breakpoint(
            self, process, "break after assignment", lldb.SBFileSpec("main.swift")
        )

        description = self.frame().FindVariable("object").GetObjectDescription()
        self.assertIsNotNone(description)
        self.assertIn("POClass:", description)
