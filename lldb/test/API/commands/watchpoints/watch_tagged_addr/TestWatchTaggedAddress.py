"""
Test LLDB can set and hit watchpoints on tagged addresses
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWatchTaggedAddresses(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Skip this test if not running on AArch64 target that supports PAC
        if not self.isAArch64PAuth():
            self.skipTest("Target must support pointer authentication.")

        # Set source filename.
        self.source = "main.c"

        # Invoke the default build rule.
        self.build()

        # Get the path of the executable
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_watch_hit_tagged_ptr_access(self):
        """
        Test that LLDB hits watchpoint installed on an untagged address with
        memory access by a tagged pointer.
        """

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_symbol(self, "main")

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped due to the breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Set the watchpoint variable declaration line number.
        self.decl = line_number(self.source, "// Watchpoint variable declaration.")

        # Now let's set a watchpoint on 'global_var'.
        self.expect(
            "watchpoint set variable global_var",
            WATCHPOINT_CREATED,
            substrs=[
                "Watchpoint created",
                "size = 4",
                "type = w",
                "%s:%d" % (self.source, self.decl),
            ],
        )

        self.verify_watch_hits()

    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_watch_set_on_tagged_ptr(self):
        """Test that LLDB can install and hit watchpoint on a tagged address"""

        # Find the line number to break inside main().
        self.line = line_number(self.source, "// Set break point at this line.")

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped due to the breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Now let's set a expression watchpoint on 'tagged_ptr'.
        self.expect(
            "watchpoint set expression -s 4 -- tagged_ptr",
            WATCHPOINT_CREATED,
            substrs=["Watchpoint created", "size = 4", "type = w"],
        )

        self.verify_watch_hits()

    def verify_watch_hits(self):
        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect(
            "watchpoint list -v",
            substrs=["Number of supported hardware watchpoints:", "hit_count = 0"],
        )

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_WATCHPOINT,
            substrs=["stop reason = watchpoint"],
        )

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_WATCHPOINT,
            substrs=["stop reason = watchpoint"],
        )

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 2.
        self.expect("watchpoint list -v", substrs=["hit_count = 2"])

        # On some hardware, during __do_global_dtors_aux a flag is set near
        # the global which can trigger the watchpoint. So we must remove it.
        self.runCmd("watchpoint delete 1")
        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status", substrs=["exited"])
