"""Test that a global ObjC object found before the process is started updates correctly."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCGlobalVar(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source = lldb.SBFileSpec("main.m")

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test that a global ObjC object found before the process is started updates correctly."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bkpt = target.BreakpointCreateBySourceRegex('NSLog', self.main_source)
        self.assertTrue(bkpt, VALID_BREAKPOINT)

        # Before we launch, make an SBValue for our global object pointer:
        g_obj_ptr = target.FindFirstGlobalVariable("g_obj_ptr")
        self.assertTrue(g_obj_ptr.GetError().Success(), "Made the g_obj_ptr")
        self.assertEqual(
            g_obj_ptr.GetValueAsUnsigned(10), 0,
            "g_obj_ptr is initially null")

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)
        if len(threads) != 1:
            self.fail("Failed to stop at breakpoint 1.")

        thread = threads[0]

        dyn_value = g_obj_ptr.GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(
            dyn_value.GetError().Success(),
            "Dynamic value is valid")
        self.assertEquals(dyn_value.GetObjectDescription(), "Some NSString")
