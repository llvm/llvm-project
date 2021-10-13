"""Test that we handle inferiors which change their process group"""



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ChangeProcessGroupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c', '// Set breakpoint here')

    @skipIfFreeBSD  # Times out on FreeBSD llvm.org/pr23731
    @skipIfWindows  # setpgid call does not exist on Windows
    @expectedFailureAndroid("http://llvm.org/pr23762", api_levels=[16])
    @expectedFailureNetBSD
    @skipIftvOS # fork not available on tvOS.
    @skipIfwatchOS # fork not available on watchOS.
    def test_setpgid(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(self,
            "pid_file_%d" % (int(time.time())))
        self.addTearDownHook(
            lambda: self.run_platform_command(
                "rm %s" %
                (pid_file_path)))

        popen = self.spawnSubprocess(exe, [pid_file_path])

        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        # make sure we cleanup the forked child also
        def cleanupChild():
            if lldb.remote_platform:
                lldb.remote_platform.Kill(int(pid))
            else:
                if os.path.exists("/proc/" + pid):
                    os.kill(int(pid), signal.SIGKILL)
        self.addTearDownHook(cleanupChild)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, int(pid), error)
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # set a breakpoint just before the setpgid() call
        lldbutil.run_break_set_by_file_and_line(
            self, 'main.c', self.line, num_expected_locations=-1)

        thread = process.GetSelectedThread()

        # release the child from its loop
        value = thread.GetSelectedFrame().EvaluateExpression("release_child_flag = 1")
        self.assertTrue(value.IsValid())
        self.assertEquals(value.GetValueAsUnsigned(0), 1)
        process.Continue()

        # make sure the child's process group id is different from its pid
        value = thread.GetSelectedFrame().EvaluateExpression("(int)getpgid(0)")
        self.assertTrue(value.IsValid())
        self.assertNotEqual(value.GetValueAsUnsigned(0), int(pid))

        # step over the setpgid() call
        thread.StepOver()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # verify that the process group has been set correctly
        # this also checks that we are still in full control of the child
        value = thread.GetSelectedFrame().EvaluateExpression("(int)getpgid(0)")
        self.assertTrue(value.IsValid())
        self.assertEqual(value.GetValueAsUnsigned(0), int(pid))

        # run to completion
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
