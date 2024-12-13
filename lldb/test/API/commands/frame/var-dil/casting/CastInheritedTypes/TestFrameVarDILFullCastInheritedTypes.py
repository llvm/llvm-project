"""
Make sure 'frame var' using DIL parser/evaultor works for C-Style casts..
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

        # TestCastDerivedToBase

        self.expect("frame variable 'static_cast<CxxA*>(&a)->a'",
                    substrs=["1"])
        self.expect("frame variable 'static_cast<CxxA*>(&c)->a'",
                    substrs=["3"])
        self.expect("frame variable 'static_cast<CxxB*>(&c)->b'",
                    substrs=["4"])
        # CAROLINE!!
        #self.expect("frame variable 'static_cast<CxxB*>(&c)->c'",
        #            error=True, substrs=["no member named 'c' in 'CxxB'"])
        self.expect("frame variable 'static_cast<CxxB*>(&e)->b'",
                    substrs=["8"])
        self.expect("frame variable 'static_cast<CxxC*>(&e)->a'",
                    substrs=["7"])
        self.expect("frame variable 'static_cast<CxxC*>(&e)->b'",
                    substrs=["8"])
        self.expect("frame variable 'static_cast<CxxC*>(&e)->c'",
                    substrs=["9"])
        self.expect("frame variable 'static_cast<CxxB*>(&d)'", error=True,
                    substrs=["static_cast from 'CxxD *' to 'CxxB *', which "
                             "are not related by inheritance, is not allowed"])

        # Cast via virtual inheritance.
        self.expect("frame variable 'static_cast<CxxA*>(&vc)->a'",
                    substrs=["12"])
        self.expect("frame variable 'static_cast<CxxB*>(&vc)->b'",
                    substrs=["13"])
        # CAROLINE!!
        #self.expect("frame variable 'static_cast<CxxB*>(&vc)->c'",
        #            error=True, substrs=["no member named 'c' in 'CxxB'"])
        self.expect("frame variable 'static_cast<CxxB*>(&ve)->b'",
                    substrs=["16"])
        self.expect("frame variable 'static_cast<CxxC*>(&ve)'", error=True,
                    substrs=["static_cast from 'CxxVE *' to 'CxxC *', which "
                             "are not related by inheritance, is not allowed"])

        # Same with references.
        self.expect("frame variable 'static_cast<CxxA&>(a).a'", substrs=["1"])
        self.expect("frame variable 'static_cast<CxxA&>(c).a'", substrs=["3"])
        self.expect("frame variable 'static_cast<CxxB&>(c).b'", substrs=["4"])
        # CAROLINE!!
        #self.expect("frame variable 'static_cast<CxxB&>(c).c'",
        #            error=True, substrs=["no member named 'c' in 'CxxB'"])
        self.expect("frame variable 'static_cast<CxxB&>(e).b'", substrs=["8"])
        self.expect("frame variable 'static_cast<CxxC&>(e).a'", substrs=["7"])
        self.expect("frame variable 'static_cast<CxxC&>(e).b'", substrs=["8"])
        self.expect("frame variable 'static_cast<CxxC&>(e).c'", substrs=["9"])
        self.expect("frame variable 'static_cast<CxxB&>(d)'", error=True,
                    substrs=["static_cast from 'CxxD' to 'CxxB &', which are "
                             "not related by inheritance, is not allowed"])

        self.expect("frame variable 'static_cast<CxxA&>(vc).a'",
                    substrs=["12"])
        self.expect("frame variable 'static_cast<CxxB&>(vc).b'",
                    substrs=["13"])
        # CAROLINE!!
        #self.expect("frame variable 'static_cast<CxxB&>(vc).c'",
        #            error=True, substrs=["no member named 'c' in 'CxxB'"])
        self.expect("frame variable 'static_cast<CxxB&>(ve).b'",
                    substrs=["16"])
        self.expect("frame variable 'static_cast<CxxC&>(ve)'", error=True,
                    substrs=["static_cast from 'CxxVE' to 'CxxC &', which are"
                             " not related by inheritance, is not allowed"])


        # TestCastBaseToDerived
        # CAROLINE!!
        #self.expect("frame variable 'static_cast<CxxE*>(e_as_b)->a'",
        #            substrs=["7"])
        self.expect("frame variable 'static_cast<CxxE*>(e_as_b)->b'",
                    substrs=["8"])
        #self.expect("frame variable 'static_cast<CxxE*>(e_as_b)->c'",
        #            substrs=["9"])
        self.expect("frame variable 'static_cast<CxxE*>(e_as_b)->d'",
                    substrs=["10"])
        self.expect("frame variable 'static_cast<CxxE*>(e_as_b)->e'",
                    substrs=["11"])

        # Same with references.
        self.expect("frame variable 'static_cast<CxxE&>(*e_as_b).a'",
                    substrs=["7"])
        self.expect("frame variable 'static_cast<CxxE&>(*e_as_b).b'",
                    substrs=["8"])
        self.expect("frame variable 'static_cast<CxxE&>(*e_as_b).c'",
                    substrs=["9"])
        self.expect("frame variable 'static_cast<CxxE&>(*e_as_b).d'",
                    substrs=["10"])
        self.expect("frame variable 'static_cast<CxxE&>(*e_as_b).e'",
                    substrs=["11"])

        # Base-to-derived conversion isn't possible for virtually inherited
        # types.
        self.expect("frame variable 'static_cast<CxxVE*>(ve_as_b)'", error=True,
                    substrs=["cannot cast 'CxxB *' to 'CxxVE *' via virtual "
                             "base 'CxxB'"])
        self.expect("frame variable 'static_cast<CxxVE&>(*ve_as_b)'", error=True,
                    substrs=["cannot cast 'CxxB' to 'CxxVE &' via virtual "
                             "base 'CxxB'"])
