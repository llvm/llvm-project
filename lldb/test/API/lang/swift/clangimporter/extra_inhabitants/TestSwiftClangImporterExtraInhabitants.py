import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftClangImporterExtraInhabitants(TestBase):
    @swiftTest
    @skipUnlessDarwin
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    def test(self):
        """Test that the extra inhabitants are correctly computed for various
           kinds of Objective-C pointers, by using them in enums."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        var = self.frame().FindVariable("mystruct")
        check = lldbutil.check_variable

        check(self, var.GetChildAtIndex(0), value="0")
        check(self, var.GetChildAtIndex(1),
              typename="Swift.Optional<Swift.OpaquePointer>",
              summary="nil")

        check(self, var.GetChildAtIndex(2), value="2")
        check(self, var.GetChildAtIndex(3),
              typename="Swift.Optional<Foo.BridgedPtr>",
              summary="nil")

        check(self, var.GetChildAtIndex(4), value="4")
        check(self, var.GetChildAtIndex(5),
              typename="Swift.Optional<Swift.AnyObject>",
              summary="nil")

        check(self, var.GetChildAtIndex(6), value="6")
        check(self, var.GetChildAtIndex(7),
              typename="Swift.Optional<Swift.UnsafeMutableRawPointer>",
              summary="nil")

        check(self, var.GetChildAtIndex(8), value="8")
