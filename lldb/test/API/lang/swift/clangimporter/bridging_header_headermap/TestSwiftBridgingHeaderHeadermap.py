# TestSwiftBridgingHeaderHeadermap.py
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
import shutil

class TestSwiftBridgingHeaderHeadermap(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.runCmd('settings set frame-format ""')
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('dylib.swift'),
                                          extra_images=['dylib'])
        self.expect("fr v -d run-target -- a",
                    substrs=['(dylib.C<a.Wrapper>.Something)', "hello"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        
