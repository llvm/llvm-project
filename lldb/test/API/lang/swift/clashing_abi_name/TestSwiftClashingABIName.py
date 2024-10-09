import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClashingABIName(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test(self):
        """Test that expressions with types in modules with clashing abi names works"""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'), 
        extra_images=['Library'])

        self.expect('expr --bind-generic-types true -- generic1', 
                    substrs=['a.Generic<a.One>', 't =', 'j = 98'])

        self.expect('expr --bind-generic-types true -- generic2', 
                    substrs=['a.Generic2<a.Generic<a.One>>', 't2 =', 't =', 'j = 98'])

        self.expect('expr --bind-generic-types true -- generic3',
                    substrs=['a.Generic2<a.Generic<a.One>>', 't2 =', 't =', 'j = 98'])
    @swiftTest
    @skipUnlessDarwin
    def test_in_self(self):
        """Test a library with a private import for which there is no debug info"""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, 'break for self', lldb.SBFileSpec('main.swift'))

        self.expect('expr --bind-generic-types true -- self', 
                    substrs=['a.Generic<a.One>', 't =', 'j = 98'])

