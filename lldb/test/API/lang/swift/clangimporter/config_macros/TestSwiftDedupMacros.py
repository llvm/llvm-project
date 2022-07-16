# TestSwiftDedupMacros.py
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

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftDedupMacros(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    # NOTE: rdar://44201206 - This test may sporadically segfault. It's likely
    # that the underlying memory corruption issue has been addressed, but due
    # to the difficulty of reproducing the crash, we are not sure. If a crash
    # is observed, try to collect a crashlog before disabling this test.
    @skipUnlessDarwin
    @swiftTest
    def testSwiftDebugMacros(self):
        """This tests that configuration macros get uniqued when building the
        scratch ast context. Note that "-D MACRO" options with a space
        are currently only combined to "-DMACRO" when they appear
        outside of the main binary.

        """
        self.build()
            
        target,  _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('dylib.swift'),
            extra_images=['Dylib'])

        # Turn on logging.
        log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f "+log)
        
        self.expect("p foo", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["42"])
        debug = 0
        space = 0
        ndebug = 0
        space_with_space = 0
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        for line in logfile:
            if "-DDEBUG=1" in line:
                debug += 1
            if "-DSPACE" in line:
                space += 1
            if " SPACE" in line:
                space_with_space += 1
            if "-UNDEBUG" in line:
                ndebug += 1
        # One extra in SwiftASTContextPerModule.
        self.assertEqual(debug, 3)
        self.assertEqual(space, 3)
        self.assertEqual(space_with_space, 0)
        self.assertEqual(ndebug, 3)
