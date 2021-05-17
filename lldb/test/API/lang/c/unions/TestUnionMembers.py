import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestUnionMembers(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_union_members(self):
        self._load_exe()

        # Set breakpoints
        bp = self.target.BreakpointCreateBySourceRegex(
            "Break here", self.src_file_spec)
        self.assertTrue(
            bp.IsValid() and bp.GetNumLocations() >= 1,
            VALID_BREAKPOINT)

        # Launch the process
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(
            self.process.GetState(), lldb.eStateStopped,
            PROCESS_STOPPED)

        thread = lldbutil.get_stopped_thread(
            self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("u")
        self.assertTrue(val.IsValid())
        val = frame.EvaluateExpression("u.s")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetNumChildren(), 2)

    def _load_exe(self):
        self.build()

        src_file = os.path.join(self.getSourceDir(), "main.c")
        self.src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(self.src_file_spec.IsValid(), "breakpoint file")

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        self.target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(self.target.IsValid(), VALID_TARGET)
