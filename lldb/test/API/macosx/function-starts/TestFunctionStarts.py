"""
Test that we read the function starts section.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

exe_name = "StripMe"  # Must match Makefile

class FunctionStartsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipUnlessDarwin
    def test_function_starts_binary(self):
        """Test that we make synthetic symbols when we have the binary."""
        self.build(dictionary={'CODESIGN': ''}) # Binary is getting stripped later.
        self.do_function_starts(False)

    @skipIfRemote
    @skipUnlessDarwin
    def test_function_starts_no_binary(self):
        """Test that we make synthetic symbols when we don't have the binary"""
        self.build(dictionary={'CODESIGN': ''}) # Binary is getting stripped later.
        self.do_function_starts(True)

    def do_function_starts(self, in_memory):
        """Run the binary, stop at our unstripped function,
           make sure the caller has synthetic symbols"""

        exe = self.getBuildArtifact(exe_name)
        # Now strip the binary, but leave externals so we can break on dont_strip_me.
        self.runBuildCommand(["strip", "-u", "-x", "-S", exe])

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(self,
            "token_pid_%d" % (int(os.getpid())))
        self.addTearDownHook(
            lambda: self.run_platform_command(
                "rm %s" %
                (pid_file_path)))

        popen = self.spawnSubprocess(exe, [pid_file_path])

        # Wait until process has fully started up.
        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        if in_memory:
          remove_file(exe)

        target = self.dbg.CreateTarget(None)
        self.assertTrue(target.IsValid(), "Got a vaid empty target.")
        error = lldb.SBError()
        attach_info = lldb.SBAttachInfo()
        attach_info.SetProcessID(popen.pid)
        attach_info.SetIgnoreExisting(False)
        process = target.Attach(attach_info, error)
        self.assertSuccess(error, "Didn't attach successfully to %d"%(popen.pid))

        bkpt = target.BreakpointCreateByName("dont_strip_me", exe)
        self.assertTrue(bkpt.GetNumLocations() > 0, "Didn't set the dont_strip_me bkpt.")

        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Didn't hit my breakpoint.")

        # Our caller frame should have been stripped.  Make sure we made a synthetic symbol
        # for it:
        thread = threads[0]
        self.assertTrue(thread.num_frames > 1, "Couldn't backtrace.")
        name = thread.frame[1].GetFunctionName()
        self.assertTrue(name.startswith("___lldb_unnamed_symbol"))



