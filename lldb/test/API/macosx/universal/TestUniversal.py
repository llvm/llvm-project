import unittest2
import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def haswellOrLater():
    features = subprocess.check_output(["sysctl", "machdep.cpu"])
    return "AVX2" in features.split()


class UniversalTestCase(TestBase):
    """Test aspects of lldb commands on universal binaries."""

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.c", "// Set break point at this line.")

    @add_test_categories(["pyapi"])
    @skipUnlessDarwin
    @unittest2.skipUnless(
        hasattr(os, "uname") and os.uname()[4] in ["x86_64"], "requires x86_64"
    )
    @skipIfDarwinEmbedded  # this test file assumes we're targetting an x86 system
    @skipIf(compiler="clang", compiler_version=["<", "7.0"])
    def test_sbdebugger_create_target_with_file_and_target_triple(self):
        """Test the SBDebugger.CreateTargetWithFileAndTargetTriple() API."""
        # Invoke the default build rule.
        self.build()

        # Note that "testit" is a universal binary.
        exe = self.getBuildArtifact("testit")

        # Create a target by the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple(
            exe, "x86_64-apple-macosx10.10"
        )
        self.assertTrue(target, VALID_TARGET)
        self.expect("image list -t -b", substrs=["x86_64-apple-macosx10.9.0 testit"])
        self.expect("target list", substrs=["testit", "arch=x86_64-apple-macosx10.10"])

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

    @skipUnlessDarwin
    @unittest2.skipUnless(
        hasattr(os, "uname") and os.uname()[4] in ["x86_64"], "requires x86_64"
    )
    @skipIfDarwinEmbedded  # this test file assumes we're targetting an x86 system
    @skipIf(compiler="clang", compiler_version=["<", "7.0"])
    def test_process_launch_for_universal(self):
        """Test process launch of a universal binary."""
        from lldbsuite.test.lldbutil import print_registers

        if not haswellOrLater():
            return

        # Invoke the default build rule.
        self.build()

        # Note that "testit" is a universal binary.
        exe = self.getBuildArtifact("testit")

        # By default, x86_64 is assumed if no architecture is specified.
        self.expect(
            "file " + exe,
            CURRENT_EXECUTABLE_SET,
            startstr="Current executable set to ",
            substrs=["testit' (x86_64h)."],
        )

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True
        )

        # We should be able to launch the x86_64h executable.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a x86_64h process launched.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.expect("image list -A -b", substrs=["x86_64h testit"])
        self.runCmd("continue")

        # Now specify x86_64 as the architecture for "testit".
        self.expect(
            "file -a x86_64 " + exe,
            CURRENT_EXECUTABLE_SET,
            startstr="Current executable set to ",
            substrs=["testit' (x86_64)."],
        )

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True
        )

        # We should be able to launch the x86_64 executable as well.
        self.runCmd("run", RUN_SUCCEEDED)

        # Check whether we have a x86_64 process launched.

        # FIXME: This wrong. We are expecting x86_64, but spawning a
        # new process currently doesn't allow specifying a *sub*-architecture.
        # <rdar://problem/46101466>
        self.expect("image list -A -b", substrs=["x86_64h testit"])
        self.runCmd("continue")

    @skipUnlessDarwin
    @unittest2.skipUnless(
        hasattr(os, "uname") and os.uname()[4] in ["x86_64"], "requires x86_64"
    )
    @skipIfDarwinEmbedded  # this test file assumes we're targetting an x86 system
    def test_process_attach_with_wrong_arch(self):
        """Test that when we attach to a binary from the wrong fork of
        a universal binary, we fix up the ABI correctly."""
        if not haswellOrLater():
            return

        # Now keep the architecture at x86_64, but switch the binary
        # we launch to x86_64h, and make sure on attach we switch to
        # the correct architecture.

        # Invoke the default build rule.
        self.build()

        # Note that "testit" is a universal binary.
        exe = self.getBuildArtifact("testit")

        # Create a target by the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple(
            exe, "x86_64-apple-macosx"
        )
        self.assertTrue(target, VALID_TARGET)
        self.expect("image list -A -b", substrs=["x86_64 testit"])

        bkpt = target.BreakpointCreateBySourceRegex("sleep", lldb.SBFileSpec("main.c"))
        self.assertTrue(bkpt.IsValid(), "Valid breakpoint")
        self.assertTrue(
            bkpt.GetNumLocations() >= 1, "Our main breakpoint has locations."
        )

        popen = self.spawnSubprocess(exe, ["keep_waiting"])

        error = lldb.SBError()
        empty_listener = lldb.SBListener()
        process = target.AttachToProcessWithID(empty_listener, popen.pid, error)
        self.assertSuccess(error, "Attached to process.")

        self.expect("image list -A -b", substrs=["x86_64h testit"])

        # It may seem odd to check the number of frames, but the bug
        # that motivated this test was that we eventually fixed the
        # architecture, but we left the ABI set to the original value.
        # In that case, if you asked the process for its architecture,
        # it would look right, but since the ABI was wrong,
        # backtracing failed.

        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]
        self.assertTrue(thread.GetNumFrames() > 1, "We were able to backtrace.")
