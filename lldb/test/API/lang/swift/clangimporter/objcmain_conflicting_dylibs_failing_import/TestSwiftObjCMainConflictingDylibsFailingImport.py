# TestSwiftObjCMainConflictingDylibsFailingImport.py
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

class TestSwiftObjCMainConflictingDylibsFailingImport(TestBase):

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
        self.build()

        target, process, _, bar_breakpoint = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Bar.swift'),
            extra_images=['Foo', 'Bar'])

        # This works because the Module-SwiftASTContext uses the dylib flags.
        self.expect("fr var bar", "expected result", substrs=["42"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        # FIXME: The following expression evaluator tests are disabled
        # because it's nondeterministic which one will work.
        
        # This initially fails with the shared scratch context and is
        # then retried with the per-dylib scratch context.
        # self.expect("p bar", "expected result", substrs=["$R0", "42"])
        # self.expect("p $R0", "expected result", substrs=["$R1", "42"])
        # self.expect("p $R1", "expected result", substrs=["$R2", "42"])
        
        # This works by accident because the search paths are in the right order.
        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        process.Continue()
        self.expect("fr var foo", "expected result", substrs=["23"])
        # FIXME: The following expression evaluator tests are disabled
        # because it's nondeterministic which one will work.

        # self.expect("p foo", "expected result", substrs=["$R3", "23"])
        # self.expect("p $R3", "expected result", substrs=["23"])
        # self.expect("p $R4", "expected result", substrs=["23"])

