import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestProcessHandle(TestBase):

    @no_debug_info_test
    @skipIfWindows
    def test_process_handle(self):
        """Test that calling process handle before we have a target, and before we
           have a process will affect the process.  Also that the signal settings
           are preserved on rerun."""
        self.build()

        # Make sure we don't accept signal values by signo with no process - we don't know what the
        # mapping will be so we can't do the right thing with bare numbers:
        lldbutil.set_actions_for_signal(self, "9", "true", None, None, expect_success=False)

        # First, I need a reference value so I can see whether changes actually took:
        (target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))
        (default_pass, default_stop, default_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV")
        
        # Let's change the value here, then exit and make sure the changed value sticks:
        new_value = "false"
        if default_pass == "true":
            new_value = "false"

        # First make sure we get an error for bogus values when running:
        lldbutil.set_actions_for_signal(self, "NOTSIGSEGV", new_value, None, None, expect_success=False)

        # Then set the one we intend to change.
        lldbutil.set_actions_for_signal(self, "SIGSEGV", new_value, None, None)

        process.Continue()
        
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
        
        # Check that we preserved the setting:
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV",from_target=True)
        self.assertEqual(curr_pass, new_value, "Pass was set correctly")
        self.assertEqual(curr_stop, "not set", "Stop was not set by us")
        self.assertEqual(curr_notify, "not set", "Notify was not set by us")

        # Run again and make sure that we prime the new process with these settings:
        process = lldbutil.run_to_breakpoint_do_run(self, target, bkpt)

        # We check the process settings now, to see what got copied into the process:
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV")
        self.assertEqual(curr_pass, new_value, "Pass was set correctly")
        self.assertEqual(curr_stop, default_stop, "Stop was its default value")
        self.assertEqual(curr_notify, default_notify, "Notify was its default value")

        # Now kill this target, set the handling and make sure the values get copied from the dummy into the new target.
        success = self.dbg.DeleteTarget(target)
        self.assertTrue(success, "Deleted the target")
        self.assertEqual(self.dbg.GetNumTargets(), 0, "We did delete all the targets.")

        # The signal settings should be back at their default - we were only setting this on the target:
        lldbutil.get_actions_for_signal(self, "SIGSEGV", from_target=True, expected_absent=True)
        # Set a valid one:
        lldbutil.set_actions_for_signal(self, "SIGSEGV", new_value, None, None)
        # Set a bogus one - we don't have a way to check pre-run so this is allowed
        # but we should get an error message when launching:
        lldbutil.set_actions_for_signal(self, "SIGNOTSIG", new_value, None, None)

        out_filename = self.getBuildArtifact('output')
        success = True
        try:
            f = open(out_filename, 'w')
        except:
            success = False

        if not success:
            self.fail("Couldn't open error output file for writing.")

        self.dbg.SetErrorFileHandle(f, False)
        # Now make a new process and make sure the right values got copied into the new target
        (target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))
        f.write("TESTPATTERN\n")
        f.flush()
        f.close()

        try:
            f = open(out_filename, 'r')
        except:
            success = False

        if not success:
            self.fail("Couldn't open error output file for reading")
        errors = f.read()
        f.close()
        
        self.assertIn("SIGNOTSIG", errors, "We warned about the unset signal")
        # Also make sure we didn't accidentally add this bogus setting to the process.
        lldbutil.set_actions_for_signal(self, "SIGNOTSIG", "true", "true", "true", expect_success=False)
        
        # Check that they went into the target:
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV",from_target=True)
        self.assertEqual(curr_pass, new_value, "Pass was set correctly")
        self.assertEqual(curr_stop, "not set", "Stop was not set by us")
        self.assertEqual(curr_notify, "not set", "Notify was not set by us")

        # And the process:
        # Check that they went into the target:
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV")
        self.assertEqual(curr_pass, new_value, "Pass was set correctly")
        self.assertEqual(curr_stop, default_stop, "Stop was its default value")
        self.assertEqual(curr_notify, default_notify, "Notify was its default value")

        # Now clear the handling, and make sure that we get the right signal values again:
        self.runCmd("process handle -c SIGSEGV")
        # Check that there is no longer configuration for SIGSEGV in the target:
        lldbutil.get_actions_for_signal(self, "SIGSEGV",from_target=True, expected_absent=True)
        # Make a new process, to make sure we did indeed reset the values:
        (target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV")
        self.assertEqual(curr_pass, new_value, "Pass was set correctly")
        self.assertEqual(curr_stop, default_stop, "Stop was its default value")
        self.assertEqual(curr_notify, default_notify, "Notify was its default value")

        # Finally remove this from the dummy target as well, and make sure it was cleared from there:
        self.runCmd("process handle -c -d SIGSEGV")
        error = process.Kill()
        self.assertSuccess(error, "Killed the process")
        success = self.dbg.DeleteTarget(target)
        self.assertTrue(success, "Destroyed the target.")
        
        (target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))
        (curr_pass, curr_stop, curr_notify) = lldbutil.get_actions_for_signal(self, "SIGSEGV")
        self.assertEqual(curr_pass, default_pass, "Pass was set correctly")
        self.assertEqual(curr_stop, default_stop, "Stop was its default value")
        self.assertEqual(curr_notify, default_notify, "Notify was its default value")
