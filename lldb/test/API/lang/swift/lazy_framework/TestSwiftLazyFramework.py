import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftLazyFramework(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test(self):
        """Test that a framework that is registered as autolinked in a Swift
           module used in the target, but not linked against the target is
           automatically loaded by LLDB."""
        self.build()
        self.expect("settings set target.swift-auto-import-frameworks true")
        self.expect("settings set target.use-all-compiler-flags true")
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

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_precise_compiler_invocation(self):
        """In modern LLDB, with precise compiler invocations the expression
           is evaluated in the local context by default."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect("expression -- 1", error=False)
