"""
Make sure 'frame var' using DIL parser/evaultor works for local variables.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILArithmetic(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT
        )

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint

        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at our breakpoint"
        )
       # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        #self.expect("settings set target.experimental.use-DIL true",
        #            substrs=[""])
        self.expect("frame variable a", substrs=["1"])
        self.expect("frame variable 'a + c'", substrs=["11"])

        self.expect("frame variable '1 + 2'", substrs=["3"])
        self.expect("frame variable '1 + 2*3'", substrs=["7"])
        self.expect("frame variable '1 + (2 - 3)'", substrs=["0"])
        self.expect("frame variable '1 == 2'", substrs=["false"])
        self.expect("frame variable '1 == 1'", substrs=["true"])

        #Note: Signed overflow is UB.
        self.expect("frame variable 'int_max + 1'",
                    substrs=["-2147483648"])
        self.expect("frame variable 'int_min - 1'", substrs=["2147483647"])
        self.expect("frame variable '2147483647 + 1'",
                    substrs=["-2147483648"])
#        self.expect("frame variable -- '-2147483648 - 1'",
#                    substrs=["2147483647"])

        self.expect("frame variable 'uint_max + 1'", substrs=["0"])
        self.expect("frame variable 'uint_zero - 1'",
                    substrs=["4294967295"])
        self.expect("frame variable '4294967295 + 1'",
                    substrs=["4294967296"])
        self.expect("frame variable '4294967295U + 1'", substrs=["0"])

        # Note: Signed overflow is UB.
        self.expect("frame variable 'll_max + 1'",
                    substrs=["-9223372036854775808"])
        self.expect("frame variable 'll_min - 1'",
                    substrs=["9223372036854775807"])
        self.expect("frame variable '9223372036854775807 + 1'",
                    substrs=["-9223372036854775808"])
#        self.expect("frame variable -- '-9223372036854775808 - 1'",
#                    patterns=["0x[0-9]+"])

        self.expect("frame variable 'ull_max + 1'", substrs=["0"])
        self.expect("frame variable 'ull_zero - 1'",
                    substrs=["18446744073709551615"])
        self.expect("frame variable '9223372036854775807 + 1'",
                    substrs=["-9223372036854775808"])
        self.expect("frame variable '9223372036854775807LL + 1'",
                    substrs=["-9223372036854775808"])
        self.expect("frame variable '18446744073709551615ULL + 1'",
                    substrs=["0"])

        # Integer literal is too large to be represented in a signed integer
        # type, interpreting as unsigned.
        self.expect("frame variable -- '-9223372036854775808'",
                    substrs=["9223372036854775808"])
        self.expect("frame variable -- '-9223372036854775808 - 1'",
                    substrs=["9223372036854775807"])
        self.expect("frame variable -- '-9223372036854775808 + 1'",
                    substrs=["9223372036854775809"])
        self.expect("frame variable -- '-9223372036854775808LL / -1'",
                    substrs=["0"])
        self.expect("frame variable -- '-9223372036854775808LL % -1'",
                    substrs=["9223372036854775808"])

        self.expect("frame variable -- '-20 / 1U'",
                    substrs=["4294967276"])
        self.expect("frame variable -- '-20LL / 1U'", substrs=["-20"])
        self.expect("frame variable -- '-20LL / 1ULL'",
                    substrs=["18446744073709551596"])

        # Unary arithmetic.
        self.expect("frame variable '+0'", substrs=["0"])
        self.expect("frame variable -- '-0'", substrs=["0"])
        self.expect("frame variable '+1'", substrs=["1"])
        self.expect("frame variable -- '-1'", substrs=["-1"])
        self.expect("frame variable 'c'", substrs=["'\\n'"])
        self.expect("frame variable '+c'", substrs=["10"])
        self.expect("frame variable -- '-c'", substrs=["-10"])
        self.expect("frame variable 'uc'", substrs=["'\\x01'"])
        self.expect("frame variable -- '-uc'", substrs=["-1"])
        self.expect("frame variable '+p'", patterns=["0x[0-9]+"])
        self.expect("frame variable -- '-p'", error=True,
                    substrs=["invalid argument type 'int *' to unary "
                             "expression"])

        # Floating tricks.
        self.expect("frame variable '+0.0'", substrs=["0"])
        self.expect("frame variable -- '-0.0'", substrs=["-0"])
        self.expect("frame variable '0.0 / 0'", substrs=["NaN"])
        self.expect("frame variable '0 / 0.0'", substrs=["NaN"])
        self.expect("frame variable '1 / +0.0'", substrs=["+Inf"])
        self.expect("frame variable '1 / -0.0'", substrs=["-Inf"])
        self.expect("frame variable '+0.0 / +0.0  != +0.0 / +0.0'",
                    substrs=["true"])
        self.expect("frame variable -- '-1.f * 0'", substrs=["-0"])
        self.expect("frame variable '0x0.123p-1'",
                    substrs=["0.0355224609375"])

        self.expect("frame variable 'fnan < fnan'", substrs=["false"])
        self.expect("frame variable 'fnan == fnan'", substrs=["false"])
        self.expect("frame variable '(unsigned int) fdenorm'",
                    substrs=["0"])
        self.expect("frame variable '(unsigned int) (1.0f + fdenorm)'",
                    substrs=["1"])

        # Invalid remainder.
        self.expect("frame variable '1.1 % 2'", error=True,
                    substrs=["invalid operands to binary expression ('double' "
                             "and 'int')"])

        #  References and typedefs.
        self.expect("frame variable 'r + 1'", substrs=["3"])
        self.expect("frame variable 'r - 1l'", substrs=["1"])
        self.expect("frame variable 'r * 2u'", substrs=["4"])
        self.expect("frame variable 'r / 2ull'", substrs=["1"])
        self.expect("frame variable 'my_r + 1'", substrs=["3"])
        self.expect("frame variable 'my_r - 1'", substrs=["1"])
        self.expect("frame variable 'my_r * 2'", substrs=["4"])
        self.expect("frame variable 'my_r / 2'", substrs=["1"])
        self.expect("frame variable 'r + my_r'", substrs=["4"])
        self.expect("frame variable 'r - my_r'", substrs=["0"])
        self.expect("frame variable 'r * my_r'", substrs=["4"])
        self.expect("frame variable 'r / my_r'", substrs=["1"])

        # Some promotions and conversions.
        self.expect("frame variable '(uint8_t)250 + (uint8_t)250'",
                    substrs=["500"])

        # Makes sure that the expression isn't parsed as two types `r<r>` and `r`.
        self.expect("frame variable '(r < r > r)'", substrs=["false"])

        #  # On Windows sizeof(int) == sizeof(long) == 4.                              #
        #  if constexpr (sizeof(int) == sizeof(long)) {
        #    self.expect("frame variable '(unsigned int)4294967295 + (long)2'", substrs=["1"])
        #    self.expect("frame variable '((unsigned int)1 + (long)1) - 3'", substrs=["4294967295"])
        #  } else {
            # On Linux sizeof(int) == 4 and sizeof(long) == 8.
        self.expect("frame variable '(unsigned int)4294967295 + (long)2'",
                    substrs=["4294967297"])
        self.expect("frame variable '((unsigned int)1 + (long)1) - 3'",
                    substrs=["-1"])
