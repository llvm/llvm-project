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


class TestSwiftDeploymentTarget(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(bugnumber="rdar://60396797", # should work but crashes.
            setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "10.11"])
    @swiftTest
    def test_swift_deployment_target(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("p f", substrs=['i = 23'])

    @skipIf(bugnumber="rdar://60396797", # should work but crashes.
            setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "10.11"])
    @swiftTest
    def test_swift_deployment_target_dlopen(self):
        self.build()
        target, process, _, _, = lldbutil.run_to_name_breakpoint(
            self, 'main', exe_name="dlopen_module")
        bkpt = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('NewerTarget.swift'))
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("p self", substrs=['i = 23'])

