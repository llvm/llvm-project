import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftLazyFramework(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_system_framework(self):
        """Test that a framework that is registered as autolinked in a Swift
           module used in the target, but not linked against the target is
           automatically loaded by LLDB."""
        self.build()
        self.expect("settings set target.swift-auto-import-frameworks true")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        
        # Verify that lazy is not linked in.
        self.runCmd("image list")
        output = self.res.GetOutput()
        self.assertIn("dyld", output)
        self.assertNotIn("Lazy.framework/Versions/A/Lazy", output)
        # FIXME: we should automatically retry the expression on dylib import.
        self.expect("expression -- 1", error=True)
        self.expect("expression -- C()", substrs=['23'])

        # Verify that lazy has been dynamically loaded.
        self.expect("image list", substrs=["Lazy.framework/Versions/A/Lazy"])
