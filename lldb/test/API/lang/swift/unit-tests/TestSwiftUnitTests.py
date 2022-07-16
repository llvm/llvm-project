# TestSwiftUnitTests.py
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
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftUnitTests(TestBase):

    @skipUnlessDarwin
    @swiftTest
    # The creation of the .xctest framework messes with the AST search path.
    @skipIf(debug_info=no_match("dsym"))
    @skipIfDarwinEmbedded # swift crash inspecting swift stdlib with little other swift loaded <rdar://problem/55079456> 
    def test_cross_module_extension(self):
        """Test that XCTest-based unit tests work"""
        self.build()
        lldbutil.run_to_source_breakpoint(self,
                                          "Set breakpoint here",
                                          lldb.SBFileSpec('xctest.c'),
                                          exe_name = "xctest")
        self.expect("expr -l Swift -- import test")
        self.expect("expr -l Swift -- doTest()",
                    substrs=['Int','$R0','=','3'])
