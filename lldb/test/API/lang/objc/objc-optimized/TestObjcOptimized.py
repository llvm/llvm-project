"""
Test that objective-c expression parser continues to work for optimized build.

Fixed a bug in the expression parser where the 'this'
or 'self' variable was not properly read if the compiler
optimized it into a register.
"""



import lldb
import re

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# rdar://problem/9087739
# test failure: objc_optimized does not work for "-C clang -A i386"


class ObjcOptimizedTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    myclass = "MyClass"
    mymethod = "description"
    method_spec = "-[%s %s]" % (myclass, mymethod)

    @expectedFailureAll(remote=True)
    def test_break(self):
        """Test 'expr member' continues to work for optimized build."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_symbol(
            self,
            self.method_spec,
            num_expected_locations=1,
            sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
            patterns=[
                "frame.*0:.*%s %s" %
                (self.myclass,
                 self.mymethod)])

        self.expect('expression member',
                    startstr="(int) $0 = 5")

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
                    substrs=[("(%s *) $1 = " % self.myclass), desired_pointer])

        self.expect('expression self->non_member', error=True,
                    substrs=["does not have a member named 'non_member'"])
