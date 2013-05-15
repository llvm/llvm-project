"""
Test that objective-c expression parser continues to work for optimized build.

http://llvm.org/viewvc/llvm-project?rev=126973&view=rev
Fixed a bug in the expression parser where the 'this'
or 'self' variable was not properly read if the compiler
optimized it into a register.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil
import re

# rdar://problem/9087739
# test failure: objc_optimized does not work for "-C clang -A i386"
@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class ObjcOptimizedTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "objc-optimized")
    myclass = "MyClass"
    mymethod = "description"
    method_spec = "-[%s %s]" % (myclass, mymethod)

    @dsym_test
    def test_break_with_dsym(self):
        """Test 'expr member' continues to work for optimized build."""
        self.buildDsym()
        self.objc_optimized()

    @dwarf_test
    def test_break_with_dwarf(self):
        """Test 'expr member' continues to work for optimized build."""
        self.buildDwarf()
        self.objc_optimized()

    def objc_optimized(self):
        """Test 'expr member' continues to work for optimized build."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_symbol (self, self.method_spec, num_expected_locations=1, sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint"],
            patterns = ["frame.*0:.*%s %s" % (self.myclass, self.mymethod)])

        self.expect('expression member',
            startstr = "(int) $0 = 5")

        # <rdar://problem/12693963>
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand('frame variable self', result)
        output = result.GetOutput()

        desired_pointer = "0x0"

        mo = re.search("0x[0-9a-f]+", output)

        if mo:
            desired_pointer = mo.group(0)

        self.expect('expression (self)',
            substrs = [("(%s *) $1 = " % self.myclass), desired_pointer])

        self.expect('expression self->non_member', error=True,
            substrs = ["does not have a member named 'non_member'"])

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
