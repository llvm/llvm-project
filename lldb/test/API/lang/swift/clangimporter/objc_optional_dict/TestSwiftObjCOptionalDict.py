import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftObjCOptionalDict(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # This should from DWRAF, without loading a Swift module.
        self.expect("settings set symbols.swift-typesystem-compiler-fallback false")
        d = self.frame().FindVariable("dict")
        lldbutil.check_variable(self, d, summary='0 key/value pairs', value='some')
