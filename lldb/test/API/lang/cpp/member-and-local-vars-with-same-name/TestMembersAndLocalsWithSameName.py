import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestMembersAndLocalsWithSameName(TestBase):
    def test_when_stopped_in_method(self):
        self._load_exe()

        # Set breakpoints
        bp1 = self.target.BreakpointCreateBySourceRegex("Break 1", self.src_file_spec)
        self.assertTrue(bp1.IsValid() and bp1.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp2 = self.target.BreakpointCreateBySourceRegex("Break 2", self.src_file_spec)
        self.assertTrue(bp2.IsValid() and bp2.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp3 = self.target.BreakpointCreateBySourceRegex("Break 3", self.src_file_spec)
        self.assertTrue(bp3.IsValid() and bp3.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp4 = self.target.BreakpointCreateBySourceRegex("Break 4", self.src_file_spec)
        self.assertTrue(bp4.IsValid() and bp4.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        self._test_globals()

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("a")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 12345)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 54321)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 34567)

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("a")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10001)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10002)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10003)

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("a")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 1)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 2)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 778899)

    def test_when_stopped_in_function(self):
        self._load_exe()

        # Set breakpoints
        bp1 = self.target.BreakpointCreateBySourceRegex("Break 1", self.src_file_spec)
        self.assertTrue(bp1.IsValid() and bp1.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp5 = self.target.BreakpointCreateBySourceRegex("Break 5", self.src_file_spec)
        self.assertTrue(bp5.IsValid() and bp5.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp6 = self.target.BreakpointCreateBySourceRegex("Break 6", self.src_file_spec)
        self.assertTrue(bp6.IsValid() and bp6.GetNumLocations() >= 1, VALID_BREAKPOINT)
        bp7 = self.target.BreakpointCreateBySourceRegex("Break 7", self.src_file_spec)
        self.assertTrue(bp7.IsValid() and bp7.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory()
        )
        self.assertTrue(self.process.IsValid(), PROCESS_IS_VALID)

        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        self._test_globals()

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        self.enable_expression_log()
        val = frame.EvaluateExpression("a")
        self.disable_expression_log_and_check_for_locals(["a"])
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 12345)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 54321)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 34567)

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("a")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10001)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10002)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 10003)

        self.enable_expression_log()
        val = frame.EvaluateExpression("c-b")
        self.disable_expression_log_and_check_for_locals(["c", "b"])
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 1)

        self.process.Continue()
        self.assertEqual(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        val = frame.EvaluateExpression("a")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 1)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 2)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 778899)

        self.enable_expression_log()
        val = frame.EvaluateExpression("a+b")
        self.disable_expression_log_and_check_for_locals(["a", "b"])
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 3)

    def _load_exe(self):
        self.build()

        cwd = os.getcwd()

        src_file = os.path.join(cwd, "main.cpp")
        self.src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(self.src_file_spec.IsValid(), "breakpoint file")

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        self.target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(self.target.IsValid(), VALID_TARGET)

    def _test_globals(self):
        thread = lldbutil.get_stopped_thread(self.process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())

        self.enable_expression_log()
        val = frame.EvaluateExpression("a")
        self.disable_expression_log_and_check_for_locals([])
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 112233)

        val = frame.EvaluateExpression("b")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 445566)

        val = frame.EvaluateExpression("c")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsUnsigned(), 778899)

    def enable_expression_log(self):
        log_file = self.getBuildArtifact("expr.log")
        self.runCmd("log enable  -f '%s' lldb expr" % (log_file))

    def disable_expression_log_and_check_for_locals(self, variables):
        log_file = self.getBuildArtifact("expr.log")
        self.runCmd("log disable lldb expr")
        local_var_regex = re.compile(r".*__lldb_local_vars::(.*);")
        matched = []
        with open(log_file, "r") as log:
            for line in log:
                if line.find("LLDB_BODY_START") != -1:
                    break
                m = re.match(local_var_regex, line)
                if m:
                    self.assertIn(m.group(1), variables)
                    matched.append(m.group(1))
        self.assertEqual([item for item in variables if item not in matched], [])
