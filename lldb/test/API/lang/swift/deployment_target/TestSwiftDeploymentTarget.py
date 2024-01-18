# TestSwiftDeploymentTarget.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldbsuite.test.lldbinline as lldbinline

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import os


class TestSwiftDeploymentTarget(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    @swiftTest
    def test_swift_deployment_target(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,
                                          "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("expression f", substrs=['i = 23'])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    @swiftTest
    def test_swift_deployment_target_dlopen(self):
        self.build()
        target, process, _, _, = lldbutil.run_to_name_breakpoint(
            self, 'main', exe_name="dlopen_module")
        bkpt = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('NewerTarget.swift'))
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("expression self", substrs=['i = 23'])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    # FIXME: This config started failing in CI only after switching to
    # the query-based FindTypes API.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    def test_swift_deployment_target_from_macho(self):
        self.build(dictionary={"MAKE_DSYM": "NO"})
        os.unlink(self.getBuildArtifact("a.swiftmodule"))
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression f", substrs=["i = 23"])
        self.filecheck('platform shell cat ""%s"' % log, __file__)
#       CHECK: SwiftASTContextForExpressions::SetTriple({{.*}}apple-macosx11.0.0
#       CHECK:  SwiftASTContextForExpressions::RegisterSectionModules("a.out") retrieved 0 AST Data blobs
