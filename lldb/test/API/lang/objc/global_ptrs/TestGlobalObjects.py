"""Test that a global ObjC object found before the process is started updates correctly."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCGlobalVar(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source = lldb.SBFileSpec("main.m")

    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test that a global ObjC object found before the process is started updates correctly."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bkpt = target.BreakpointCreateBySourceRegex("NSLog", self.main_source)
        self.assertTrue(bkpt, VALID_BREAKPOINT)

        # Before we launch, make an SBValue for our global object pointer:
        g_obj_ptr = target.FindFirstGlobalVariable("g_obj_ptr")
        self.assertSuccess(g_obj_ptr.GetError(), "Made the g_obj_ptr")
        self.assertEqual(
            g_obj_ptr.GetValueAsUnsigned(10), 0, "g_obj_ptr is initially null"
        )

        lldbutil.run_to_breakpoint_do_run(self, target, bkpt)

        dyn_value = g_obj_ptr.GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(dyn_value.GetError().Success(), "Dynamic value is valid")
        self.assertEqual(dyn_value.GetObjectDescription(), "Some NSString")
