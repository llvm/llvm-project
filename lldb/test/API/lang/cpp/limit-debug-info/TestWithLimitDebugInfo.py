import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWithLimitDebugInfo(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["dwarf", "dwo"])
    def test_limit_debug_info(self):
        self.build()

        src_file = os.path.join(self.getSourceDir(), "main.cpp")
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "breakpoint file")

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        breakpoint = target.BreakpointCreateBySourceRegex(
            "break here", src_file_spec)
        self.assertTrue(
            breakpoint.IsValid() and breakpoint.GetNumLocations() >= 1,
            VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertEqual(
            process.GetState(), lldb.eStateStopped,
            PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        thread.StepInto()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        self.expect_expr("1", result_type="int", result_value="1")

        v2 = frame.EvaluateExpression("this")
        self.assertTrue(
            v2.IsValid(),
            "'expr this' results in a valid SBValue object")
        self.assertSuccess(
            v2.GetError(),
            "'expr this' succeeds without an error.")
