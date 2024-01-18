# TestSwiftPathWithColons.py
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
Test that LLDB correctly handles paths with colons
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import shutil
import unittest2


# this should be a perfectly general feature but I could not
# cause the failure to reproduce against clang, so put it here
class TestSwiftPathWithColon(TestBase):
    @skipIf(oslist=['windows'])
    @skipIfiOSSimulator
    @swiftTest
    def test_path_with_colon(self):
        """Test that LLDB correctly handles paths with colons"""
        self.do_test()

    def do_test(self):
        """Test that LLDB correctly handles paths with colons"""
        src = os.path.join(self.getSourceDir(), 'main.swift')
        colon_dir = self.getBuildArtifact('pro:ject')
        copied_src = os.path.join(colon_dir, 'main.swift')
        dst = os.path.join(colon_dir, 'a.out')
        dst_makefile = os.path.join(colon_dir, 'Makefile')
        exe = os.path.join(colon_dir, 'a.out')

        if not os.path.exists(colon_dir):
            os.makedirs(colon_dir)

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            shutil.rmtree(colon_dir)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        f = open(dst_makefile, 'w')
        f.write('''
SWIFT_SOURCES := main.swift
include Makefile.rules
''')
        f.close()

        shutil.copy(src, copied_src)

        self.build()

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Don't allow ansi highlighting to interfere with the output.
        self.runCmd('settings set stop-show-column none')

        self.expect('breakpoint set -l 13', substrs=['foo'])

        self.expect('source list -l 13', substrs=['return x + y - z'])

        self.expect('run', substrs=['return x + y - z'])

