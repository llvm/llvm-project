"""
Test that dynamically discovered ivars of type IMP do not crash LLDB
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCiVarIMPTestCase(TestBase):
    @skipIf(archs=["i386"])  # objc file does not build for i386
    @no_debug_info_test
    def test_imp_ivar_type(self):
        """Test that dynamically discovered ivars of type IMP do not crash LLDB"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("repro.m")
        )

        self.expect(
            "frame variable --ptr-depth=1 --show-types -d run -- object",
            substrs=["(MyClass *) object = 0x", "(void *) myImp = 0x"],
        )
        self.expect(
            "disassemble --start-address `((MyClass*)object)->myImp`",
            substrs=["-[MyClass init]"],
        )
