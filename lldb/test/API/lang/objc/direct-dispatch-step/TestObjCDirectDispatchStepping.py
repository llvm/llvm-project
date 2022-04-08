"""Test stepping through ObjC method dispatch in various forms."""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCDirectDispatchStepping(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = lldb.SBFileSpec("stepping-tests.m")

    @add_test_categories(['pyapi', 'basic_process'])
    @expectedFailureAll(remote=True)
    def test_with_python_api(self):
        """Test stepping through the 'direct dispatch' optimized method calls."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Stop here to start stepping",
                                                                            self.main_source)
        stop_bkpt = target.BreakpointCreateBySourceRegex("// Stop Location [0-9]+", self.main_source)
        self.assertEqual(stop_bkpt.GetNumLocations(), 15)
                                                         
        # Here we step through all the overridden methods of OverridesALot
        # The last continue will get us to the call ot OverridesInit.
        for idx in range(2,16):
            thread.StepInto()
            func_name = thread.GetFrameAtIndex(0).GetFunctionName()
            self.assertIn("OverridesALot", func_name, "%d'th step did not match name: %s"%(idx, func_name))
            stop_threads = lldbutil.continue_to_breakpoint(process, stop_bkpt)
            self.assertEqual(len(stop_threads), 1)
            self.assertEqual(stop_threads[0], thread)

        thread.StepInto()
        func_name = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertEqual(func_name, "-[OverridesInit init]", "Stopped in [OverridesInit init]")
        

            
