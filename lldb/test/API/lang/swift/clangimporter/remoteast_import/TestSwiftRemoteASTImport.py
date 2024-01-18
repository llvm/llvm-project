# TestSwiftRemoteASTImport.py
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

class TestSwiftRemoteASTImport(TestBase):
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def testSwiftRemoteASTImport(self):
        """This tests that RemoteAST querying the dynamic type of a variable
        doesn't import any modules into a module SwiftASTContext that
        weren't imported by that module in the source code.

        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('Library.swift'),
                                          extra_images=['Library'])
        # FIXME: Reversing the order of these two commands does not work!
        self.expect("expr -d no-dynamic-values -- input",
                    substrs=['(Library.LibraryProtocol) $R0'])
        self.expect("expr -d run-target -- input",
                    substrs=['(a.FromMainModule) $R1'])
