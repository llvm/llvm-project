"""Test printing ObjC objects that use unbacked properties - so that the static ivar offsets are incorrect."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCIvarOffsets(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.stop_line = line_number(
            self.main_source, '// Set breakpoint here.')

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test printing ObjC objects that use unbacked properties"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(
            self.main_source, self.stop_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, "Created a process.")
        self.assertEqual(
            process.GetState(), lldb.eStateStopped,
            "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)
        self.assertEquals(len(thread_list), 1)
        thread = thread_list[0]

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "frame 0 is valid")

        mine = thread.GetFrameAtIndex(0).FindVariable("mine")
        self.assertTrue(mine, "Found local variable mine.")

        # Test the value object value for BaseClass->_backed_int

        error = lldb.SBError()

        mine_backed_int = mine.GetChildMemberWithName("_backed_int")
        self.assertTrue(
            mine_backed_int,
            "Found mine->backed_int local variable.")
        backed_value = mine_backed_int.GetValueAsSigned(error)
        self.assertSuccess(error)
        self.assertEquals(backed_value, 1111)

        # Test the value object value for DerivedClass->_derived_backed_int

        mine_derived_backed_int = mine.GetChildMemberWithName(
            "_derived_backed_int")
        self.assertTrue(mine_derived_backed_int,
                        "Found mine->derived_backed_int local variable.")
        derived_backed_value = mine_derived_backed_int.GetValueAsSigned(error)
        self.assertSuccess(error)
        self.assertEquals(derived_backed_value, 3333)

        # Make sure we also get bit-field offsets correct:

        mine_flag2 = mine.GetChildMemberWithName("flag2")
        self.assertTrue(mine_flag2, "Found mine->flag2 local variable.")
        flag2_value = mine_flag2.GetValueAsUnsigned(error)
        self.assertSuccess(error)
        self.assertEquals(flag2_value, 7)
