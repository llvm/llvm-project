import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftSubmoduleImport(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test imports of Clang submodules"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.swift'))
        # Without the import of A.B this getter would be invisible
        # because it returns a type form that submodule.
        self.expect("expression -- a.priv", substrs=['23'])
