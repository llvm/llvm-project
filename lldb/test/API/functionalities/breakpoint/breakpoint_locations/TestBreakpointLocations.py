"""
Test breakpoint commands for a breakpoint ID with multiple locations.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointLocationsTestCase(TestBase):
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_enable(self):
        """Test breakpoint enable/disable for a breakpoint ID with multiple locations."""
        self.build()
        self.breakpoint_locations_test()

    def test_shadowed_cond_options(self):
        """Test that options set on the breakpoint and location behave correctly."""
        self.build()
        self.shadowed_bkpt_cond_test()

    def test_shadowed_command_options(self):
        """Test that options set on the breakpoint and location behave correctly."""
        self.build()
        self.shadowed_bkpt_command_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.c", "// Set break point at this line.")

    def set_breakpoint(self):
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, "Target %s is not valid" % (exe))

        # This should create a breakpoint with 3 locations.

        bkpt = target.BreakpointCreateByLocation("main.c", self.line)

        # The breakpoint list should show 3 locations.
        self.assertEqual(bkpt.GetNumLocations(), 3, "Wrong number of locations")

        self.expect(
            "breakpoint list -f",
            "Breakpoint locations shown correctly",
            substrs=[
                "1: file = 'main.c', line = %d, exact_match = 0, locations = 3"
                % self.line
            ],
            patterns=[
                "where = a.out`func_inlined .+unresolved, hit count = 0",
                "where = a.out`main .+\[inlined\].+unresolved, hit count = 0",
            ],
        )

        return bkpt

    def shadowed_bkpt_cond_test(self):
        """Test that options set on the breakpoint and location behave correctly."""
        # Breakpoint option propagation from bkpt to loc used to be done the first time
        # a breakpoint location option was specifically set.  After that the other options
        # on that location would stop tracking the breakpoint.  That got fixed, and this test
        # makes sure only the option touched is affected.

        bkpt = self.set_breakpoint()
        bkpt_cond = "1 == 0"
        bkpt.SetCondition(bkpt_cond)
        self.assertEqual(bkpt.GetCondition(), bkpt_cond, "Successfully set condition")
        self.assertEqual(
            bkpt.location[0].GetCondition(),
            bkpt.GetCondition(),
            "Conditions are the same",
        )

        # Now set a condition on the locations, make sure that this doesn't effect the bkpt:
        bkpt_loc_1_cond = "1 == 1"
        bkpt.location[0].SetCondition(bkpt_loc_1_cond)
        self.assertEqual(
            bkpt.location[0].GetCondition(),
            bkpt_loc_1_cond,
            "Successfully changed location condition",
        )
        self.assertNotEqual(
            bkpt.GetCondition(),
            bkpt_loc_1_cond,
            "Changed location changed Breakpoint condition",
        )
        self.assertEqual(
            bkpt.location[1].GetCondition(),
            bkpt_cond,
            "Changed another location's condition",
        )

        # Now make sure that setting one options doesn't fix the value of another:
        bkpt.SetIgnoreCount(10)
        self.assertEqual(bkpt.GetIgnoreCount(), 10, "Set the ignore count successfully")
        self.assertEqual(
            bkpt.location[0].GetIgnoreCount(),
            10,
            "Location doesn't track top-level bkpt.",
        )

        # Now make sure resetting the condition to "" resets the tracking:
        bkpt.location[0].SetCondition("")
        bkpt_new_cond = "1 == 3"
        bkpt.SetCondition(bkpt_new_cond)
        self.assertEqual(
            bkpt.location[0].GetCondition(),
            bkpt_new_cond,
            "Didn't go back to tracking condition",
        )

        # Test that set/get accessor methods on BreakpointLocation behave correctly.
        bkpt_loc = bkpt.GetLocationAtIndex(0)

        value = "MyQueue"
        bkpt_loc.SetQueueName(value)
        self.assertEqual(
            bkpt_loc.GetQueueName(), value, "Successfully set/get bp location QueueName"
        )

        value = 5
        bkpt_loc.SetThreadID(value)
        self.assertEqual(
            bkpt_loc.GetThreadID(), value, "Successfully set/get bp location ThreadID"
        )

        value = "1 == 0"
        bkpt_loc.SetCondition(value)
        self.assertEqual(
            bkpt_loc.GetCondition(), value, "Successfully set/get bp location Condition"
        )

        value = 6
        bkpt_loc.SetThreadIndex(value)
        self.assertEqual(
            bkpt_loc.GetThreadIndex(),
            value,
            "Successfully set/get bp location ThreadIndex",
        )

        value = "MyThread"
        bkpt_loc.SetThreadName(value)
        self.assertEqual(
            bkpt_loc.GetThreadName(),
            value,
            "Successfully set/get bp location ThreadName",
        )

        value = 5
        bkpt_loc.SetIgnoreCount(value)
        self.assertEqual(
            bkpt_loc.GetIgnoreCount(),
            value,
            "Successfully set/get bp location IgnoreCount",
        )

        for value in [True, False]:
            bkpt_loc.SetAutoContinue(value)
            self.assertEqual(
                bkpt_loc.GetAutoContinue(),
                value,
                "Successfully set/get bp location AutoContinue",
            )

        for value in [True, False]:
            bkpt_loc.SetEnabled(value)
            self.assertEqual(
                bkpt_loc.IsEnabled(),
                value,
                "Successfully set/get bp location SetEnabled",
            )

        # test set/get CommandLineCommands
        set_cmds = lldb.SBStringList()
        set_cmds.AppendString("frame var")
        set_cmds.AppendString("bt")
        bkpt_loc.SetCommandLineCommands(set_cmds)

        get_cmds = lldb.SBStringList()
        bkpt_loc.GetCommandLineCommands(get_cmds)
        self.assertEqual(
            set_cmds.GetSize(), get_cmds.GetSize(), "Size of command line commands"
        )
        for idx, _ in enumerate(set_cmds):
            self.assertEqual(
                set_cmds.GetStringAtIndex(idx),
                get_cmds.GetStringAtIndex(idx),
                "Command %d" % (idx),
            )

    def shadowed_bkpt_command_test(self):
        """Test that options set on the breakpoint and location behave correctly."""
        # Breakpoint option propagation from bkpt to loc used to be done the first time
        # a breakpoint location option was specifically set.  After that the other options
        # on that location would stop tracking the breakpoint.  That got fixed, and this test
        # makes sure only the option touched is affected.

        bkpt = self.set_breakpoint()
        commands = ["AAAAAA", "BBBBBB", "CCCCCC"]
        str_list = lldb.SBStringList()
        str_list.AppendList(commands, len(commands))

        bkpt.SetCommandLineCommands(str_list)
        cmd_list = lldb.SBStringList()
        bkpt.GetCommandLineCommands(cmd_list)
        list_size = str_list.GetSize()
        self.assertEqual(
            cmd_list.GetSize(), list_size, "Added the right number of commands"
        )
        for i in range(0, list_size):
            self.assertEqual(
                str_list.GetStringAtIndex(i),
                cmd_list.GetStringAtIndex(i),
                "Mismatched commands.",
            )

        commands = ["DDDDDD", "EEEEEE", "FFFFFF", "GGGGGG"]
        loc_list = lldb.SBStringList()
        loc_list.AppendList(commands, len(commands))
        bkpt.location[1].SetCommandLineCommands(loc_list)
        loc_cmd_list = lldb.SBStringList()
        bkpt.location[1].GetCommandLineCommands(loc_cmd_list)

        loc_list_size = loc_list.GetSize()

        # Check that the location has the right commands:
        self.assertEqual(
            loc_cmd_list.GetSize(),
            loc_list_size,
            "Added the right number of commands to location",
        )
        for i in range(0, loc_list_size):
            self.assertEqual(
                loc_list.GetStringAtIndex(i),
                loc_cmd_list.GetStringAtIndex(i),
                "Mismatched commands.",
            )

        # Check that we didn't mess up the breakpoint level commands:
        self.assertEqual(
            cmd_list.GetSize(), list_size, "Added the right number of commands"
        )
        for i in range(0, list_size):
            self.assertEqual(
                str_list.GetStringAtIndex(i),
                cmd_list.GetStringAtIndex(i),
                "Mismatched commands.",
            )

        # And check we didn't mess up another location:
        untouched_loc_cmds = lldb.SBStringList()
        bkpt.location[0].GetCommandLineCommands(untouched_loc_cmds)
        self.assertEqual(untouched_loc_cmds.GetSize(), 0, "Changed the wrong location")

    def breakpoint_locations_test(self):
        """Test breakpoint enable/disable for a breakpoint ID with multiple locations."""
        self.set_breakpoint()

        # The 'breakpoint disable 3.*' command should fail gracefully.
        self.expect(
            "breakpoint disable 3.*",
            "Disabling an invalid breakpoint should fail gracefully",
            error=True,
            startstr="error: '3' is not a valid breakpoint ID.",
        )

        # The 'breakpoint disable 1.*' command should disable all 3 locations.
        self.expect(
            "breakpoint disable 1.*",
            "All 3 breakpoint locatons disabled correctly",
            startstr="3 breakpoints disabled.",
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should not stopped on any breakpoint at all.
        self.expect(
            "process status",
            "No stopping on any disabled breakpoint",
            patterns=["^Process [0-9]+ exited with status = 0"],
        )

        # The 'breakpoint enable 1.*' command should enable all 3 breakpoints.
        self.expect(
            "breakpoint enable 1.*",
            "All 3 breakpoint locatons enabled correctly",
            startstr="3 breakpoints enabled.",
        )

        # The 'breakpoint disable 1.1' command should disable 1 location.
        self.expect(
            "breakpoint disable 1.1",
            "1 breakpoint locatons disabled correctly",
            startstr="1 breakpoints disabled.",
        )

        # Run the program again.  We should stop on the two breakpoint
        # locations.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # Continue the program, there should be another stop.
        self.runCmd("process continue")

        # Stopped again.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # At this point, 1.1 has a hit count of 0 and the other a hit count of
        # 1".
        lldbutil.check_breakpoint(
            self,
            bpno=1,
            expected_locations=3,
            expected_resolved_count=2,
            expected_hit_count=2,
        )
        lldbutil.check_breakpoint(
            self,
            bpno=1,
            location_id=1,
            expected_location_resolved=False,
            expected_location_hit_count=0,
        )
        lldbutil.check_breakpoint(
            self,
            bpno=1,
            location_id=2,
            expected_location_resolved=True,
            expected_location_hit_count=1,
        )
        lldbutil.check_breakpoint(
            self,
            bpno=1,
            location_id=3,
            expected_location_resolved=True,
            expected_location_hit_count=1,
        )
