# TestSwiftDebugPrefixMap.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that LLDB correctly finds source when debug info is remapped.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import shutil
import unittest2


class TestSwiftDebugPrefixMap(TestBase):
    @swiftTest
    def test_debug_prefix_map(self):
        self.do_test()

    def do_test(self):
        # Mirror the same source tree layout used in the Makefile. When lldb is
        # invoked in the CWD, it should find the source files with the same
        # relative paths used during compilation because the compiler's CWD was
        # remapped to ".".
        src = os.path.join(self.getSourceDir(), 'Inputs', 'main.swift')
        local_srcroot = self.getBuildArtifact('srcroot')
        local_main = os.path.join(local_srcroot, 'main.swift')

        if not os.path.exists(local_srcroot):
            os.makedirs(local_srcroot)
        shutil.copy(src, local_main)

        self.build()
        # Map "." back to the build dir.
        self.expect('settings set target.source-map . ' +
                    self.getBuildArtifact("."))

        # Create the target.
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.assertTrue(target, VALID_TARGET)

        # Don't allow ANSI highlighting to interfere with the output.
        self.runCmd('settings set stop-show-column none')
        self.expect('breakpoint set -l 13', substrs=['foo'])
        self.expect('source list -l 13', substrs=['return x + y - z'])
        self.expect('run', substrs=['return x + y - z'])
