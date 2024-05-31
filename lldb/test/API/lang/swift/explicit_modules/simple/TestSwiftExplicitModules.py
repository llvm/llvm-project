import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftExplicitModules(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test explicit Swift modules"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect("expression c", substrs=['hello explicit'])

    @swiftTest
    @skipUnlessDarwin
    def test_import(self):
        """Test an implicit import inside an explicit build"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    error=True)
        self.expect("expression import Foundation")
        self.expect('expression URL(string: "https://lldb.llvm.org")',
                    substrs=["https://lldb.llvm.org"])
