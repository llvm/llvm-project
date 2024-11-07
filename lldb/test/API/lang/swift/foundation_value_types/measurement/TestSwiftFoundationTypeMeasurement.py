# TestSwiftFoundationTypeMeasurement.py
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
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIf(macos_version=["=", "15.0"])
class TestCase(TestBase):
    @swiftTest
    @skipUnlessDarwin
    @expectedFailureAll(
        bugnumber="rdar://60396797",
        setting=("symbols.use-swift-clangimporter", "false"),
    )
    def test_measurement(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expr -- measurement", substrs=["1.25 m"])

    @swiftTest
    @skipUnlessDarwin
    def test_measurement_without_swift_ast_context(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        # This setting is to test that measurements can be printed using
        # TypeSystemSwiftTypeRef only. If the SwiftASTContext fallback were
        # enabled, it can have the unwatned effect of suppressing failures
        # within the TypeRef type system.
        self.runCmd("settings set symbols.swift-enable-ast-context false")

        self.expect("frame variable measurement", substrs=["1.25 m"])
