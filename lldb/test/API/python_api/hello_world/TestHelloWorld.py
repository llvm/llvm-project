"""Test Python APIs for target (launch and attach), breakpoint, and process."""



import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class HelloWorldTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find a couple of the line numbers within main.c.
        self.line1 = line_number('main.c', '// Set break point at this line.')
        self.line2 = line_number('main.c', '// Waiting to be attached...')

    def tearDown(self):
        # Destroy process before TestBase.tearDown()
        self.dbg.GetSelectedTarget().GetProcess().Destroy()
        # Call super's tearDown().
        TestBase.tearDown(self)

    @add_test_categories(['pyapi'])
    @skipIfiOSSimulator
    def test_with_process_launch_api(self):
        """Create target, breakpoint, launch a process, and then kill it."""
        # Get the full path to our executable to be attached/debugged.
        exe = '%s_%d'%(self.getBuildArtifact(self.testMethodName), os.getpid())
        d = {'EXE': exe}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.dbg.CreateTarget(exe)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line1)

        # The default state after breakpoint creation should be enabled.
        self.assertTrue(breakpoint.IsEnabled(),
                        "Breakpoint should be enabled after creation")

        breakpoint.SetEnabled(False)
        self.assertTrue(not breakpoint.IsEnabled(),
                        "Breakpoint.SetEnabled(False) works")

        breakpoint.SetEnabled(True)
        self.assertTrue(breakpoint.IsEnabled(),
                        "Breakpoint.SetEnabled(True) works")

        # rdar://problem/8364687
        # SBTarget.Launch() issue (or is there some race condition)?

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        # The following isn't needed anymore, rdar://8364687 is fixed.
        #
        # Apply some dances after LaunchProcess() in order to break at "main".
        # It only works sometimes.
        #self.breakAfterLaunch(process, "main")

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # The breakpoint should have a hit count of 1.
        self.assertEqual(breakpoint.GetHitCount(), 1, BREAKPOINT_HIT_ONCE)

    @add_test_categories(['pyapi'])
    @skipIfiOSSimulator
    @expectedFailureNetBSD
    @skipIfReproducer # File synchronization is not supported during replay.
    def test_with_attach_to_process_with_id_api(self):
        """Create target, spawn a process, and attach to it with process id."""
        exe = '%s_%d'%(self.testMethodName, os.getpid())
        d = {'EXE': exe}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.dbg.CreateTarget(self.getBuildArtifact(exe))

        # Spawn a new process
        token = exe+'.token'
        if not lldb.remote_platform:
            token = self.getBuildArtifact(token)
            if os.path.exists(token):
                os.remove(token)
        popen = self.spawnSubprocess(self.getBuildArtifact(exe), [token])
        lldbutil.wait_for_file_on_target(self, token)

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # Let's check the stack traces of the attached process.
        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
                    substrs=['main.c:%d' % self.line2,
                             '(int)argc=2'])

    @add_test_categories(['pyapi'])
    @skipIfiOSSimulator
    @skipIfAsan # FIXME: Hangs indefinitely.
    @expectedFailureNetBSD
    @skipIfReproducer # FIXME: Unexpected packet during (active) replay
    def test_with_attach_to_process_with_name_api(self):
        """Create target, spawn a process, and attach to it with process name."""
        exe = '%s_%d'%(self.testMethodName, os.getpid())
        d = {'EXE': exe}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        target = self.dbg.CreateTarget(self.getBuildArtifact(exe))

        # Spawn a new process.
        token = exe+'.token'
        if not lldb.remote_platform:
            token = self.getBuildArtifact(token)
            if os.path.exists(token):
                os.remove(token)
        popen = self.spawnSubprocess(self.getBuildArtifact(exe), [token])
        lldbutil.wait_for_file_on_target(self, token)

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        # Pass 'False' since we don't want to wait for new instance of
        # "hello_world" to be launched.
        name = os.path.basename(exe)

        # While we're at it, make sure that passing a None as the process name
        # does not hang LLDB.
        target.AttachToProcessWithName(listener, None, False, error)
        # Also boundary condition test ConnectRemote(), too.
        target.ConnectRemote(listener, None, None, error)

        process = target.AttachToProcessWithName(listener, name, False, error)
        self.assertSuccess(error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Verify that after attach, our selected target indeed matches name.
        self.expect(
            self.dbg.GetSelectedTarget().GetExecutable().GetFilename(),
            exe=False,
            startstr=name)

        # Let's check the stack traces of the attached process.
        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False,
                    substrs=['main.c:%d' % self.line2,
                             '(int)argc=2'])
