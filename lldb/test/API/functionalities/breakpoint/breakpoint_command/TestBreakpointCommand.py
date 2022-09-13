"""
Test lldb breakpoint command add/list/delete.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import side_effect



class BreakpointCommandTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test_breakpoint_command_sequence(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.build()
        self.breakpoint_command_sequence()

    @skipIf(oslist=["windows"], bugnumber="llvm.org/pr44431")
    def test_script_parameters(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.build()
        self.breakpoint_command_script_parameters()

    def test_commands_on_creation(self):
        self.build()
        self.breakpoint_commands_on_creation()

    @skipIf(oslist=["windows"])
    @no_debug_info_test
    def test_breakpoints_with_relative_path_line_tables(self):
        """
            Test that we can set breakpoints using a full or partial path when
            line tables in the debug information has relative paths where the
            relative path is either fully contained in the specified path, or if
            the specified path also a relative path that is shorter than the
            path in the debug info.

            The "relative.yaml" contains a line table that is:

            Line table for a/b/c/main.cpp in `a.out
            0x0000000100003f94: a/b/c/main.cpp:1
            0x0000000100003fb0: a/b/c/main.cpp:2:3
            0x0000000100003fb8: a/b/c/main.cpp:2:3

            So the line table contains relative paths. We should be able to set
            breakpoints with any source path that matches this path which
            includes paths that are longer than "a/b/c/main.cpp", but also any
            relative path that is shorter than this while all specified relative
            path components still match.
        """
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "relative.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("a.out")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        # We now have debug information with line table paths that start are
        # "./a/b/c/main.cpp".

        # Make sure that all of the following paths create a breakpoint
        # successfully. We have paths that are longer than our path, and also
        # that are shorter where all relative directories still match.
        valid_paths = [
            "/x/a/b/c/main.cpp",
            "/x/y/a/b/c/main.cpp",
            "./x/y/a/b/c/main.cpp",
            "x/y/a/b/c/main.cpp",
            "./y/a/b/c/main.cpp",
            "y/a/b/c/main.cpp",
            "./a/b/c/main.cpp",
            "a/b/c/main.cpp",
            "./b/c/main.cpp",
            "b/c/main.cpp",
            "./c/main.cpp",
            "c/main.cpp",
            "./main.cpp",
            "main.cpp",
        ]
        for path in valid_paths:
            bkpt = target.BreakpointCreateByLocation(path, 2)
            self.assertTrue(bkpt.GetNumLocations() > 0,
                'Couldn\'t resolve breakpoint using full path "%s" in executate "%s" with '
                'debug info that has relative path with matching suffix' % (path, self.getBuildArtifact("a.out")))
        invalid_paths = [
            "/x/b/c/main.cpp",
            "/x/c/main.cpp",
            "/x/main.cpp",
            "./x/y/a/d/c/main.cpp",
        ]
        for path in invalid_paths:
            bkpt = target.BreakpointCreateByLocation(path, 2)
            self.assertTrue(bkpt.GetNumLocations() == 0,
                'Incorrectly resolved a breakpoint using full path "%s" with '
                'debug info that has relative path with matching suffix' % (path))


    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def test_delete_all_breakpoints(self):
        """Test that deleting all breakpoints works."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_symbol(self, "main")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("breakpoint delete")
        self.runCmd("process continue")
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* exited with status = 0'])


    def breakpoint_command_sequence(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add three breakpoints on the same line.  The first time we don't specify the file,
        # since the default file is the one containing main:
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        # Breakpoint 4 - set at the same location as breakpoint 1 to test
        # setting breakpoint commands on two breakpoints at a time
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1, loc_exact=True)
        # Make sure relative path source breakpoints work as expected. We test
        # with partial paths with and without "./" prefixes.
        lldbutil.run_break_set_by_file_and_line(
            self, "./main.c", self.line,
            num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "breakpoint_command/main.c", self.line,
            num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "./breakpoint_command/main.c", self.line,
            num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "breakpoint/breakpoint_command/main.c", self.line,
            num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "./breakpoint/breakpoint_command/main.c", self.line,
            num_expected_locations=1, loc_exact=True)
        # Test relative breakpoints with incorrect paths and make sure we get
        # no breakpoint locations
        lldbutil.run_break_set_by_file_and_line(
            self, "invalid/main.c", self.line,
            num_expected_locations=0, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "./invalid/main.c", self.line,
            num_expected_locations=0, loc_exact=True)
        # Now add callbacks for the breakpoints just created.
        self.runCmd(
            "breakpoint command add -s command -o 'frame variable --show-types --scope' 1 4")
        self.runCmd(
            "breakpoint command add -s python -o 'import side_effect; side_effect.one_liner = \"one liner was here\"' 2")

        import side_effect
        self.runCmd("command script import --allow-reload ./bktptcmd.py")

        self.runCmd(
            "breakpoint command add --python-function bktptcmd.function 3")

        # Check that the breakpoint commands are correctly set.

        # The breakpoint list now only contains breakpoint 1.
        self.expect(
            "breakpoint list", "Breakpoints 1 & 2 created", substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1" %
                self.line], patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1" %
                self.line])

        self.expect(
            "breakpoint list -f",
            "Breakpoints 1 & 2 created",
            substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1" %
                self.line],
            patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1" %
                self.line,
                "1.1: .+at main.c:%d:?[0-9]*, .+unresolved, hit count = 0" %
                self.line,
                "2.1: .+at main.c:%d:?[0-9]*, .+unresolved, hit count = 0" %
                self.line])

        self.expect("breakpoint command list 1", "Breakpoint 1 command ok",
                    substrs=["Breakpoint commands:",
                             "frame variable --show-types --scope"])
        self.expect("breakpoint command list 2", "Breakpoint 2 command ok",
                    substrs=["Breakpoint commands (Python):",
                             "import side_effect",
                             "side_effect.one_liner"])
        self.expect("breakpoint command list 3", "Breakpoint 3 command ok",
                    substrs=["Breakpoint commands (Python):",
                             "bktptcmd.function(frame, bp_loc, internal_dict)"])

        self.expect("breakpoint command list 4", "Breakpoint 4 command ok",
                    substrs=["Breakpoint commands:",
                             "frame variable --show-types --scope"])

        self.runCmd("breakpoint delete 4")

        # Next lets try some other breakpoint kinds.  First break with a regular expression
        # and then specify only one file.  The first time we should get two locations,
        # the second time only one:

        lldbutil.run_break_set_by_regexp(
            self, r"._MyFunction", num_expected_locations=2)

        lldbutil.run_break_set_by_regexp(
            self,
            r"._MyFunction",
            extra_options="-f a.c",
            num_expected_locations=1)

        lldbutil.run_break_set_by_regexp(
            self,
            r"._MyFunction",
            extra_options="-f a.c -f b.c",
            num_expected_locations=2)

        # Now try a source regex breakpoint:
        lldbutil.run_break_set_by_source_regexp(
            self,
            r"is about to return [12]0",
            extra_options="-f a.c -f b.c",
            num_expected_locations=2)

        lldbutil.run_break_set_by_source_regexp(
            self,
            r"is about to return [12]0",
            extra_options="-f a.c",
            num_expected_locations=1)

        # Reset our canary variables and run the program.
        side_effect.one_liner = None
        side_effect.bktptcmd = None
        self.runCmd("run", RUN_SUCCEEDED)

        # Check the value of canary variables.
        self.assertEquals("one liner was here", side_effect.one_liner)
        self.assertEquals("function was here", side_effect.bktptcmd)

        # Finish the program.
        self.runCmd("process continue")

        # Remove the breakpoint command associated with breakpoint 1.
        self.runCmd("breakpoint command delete 1")

        # Remove breakpoint 2.
        self.runCmd("breakpoint delete 2")

        self.expect(
            "breakpoint command list 1",
            startstr="Breakpoint 1 does not have an associated command.")
        self.expect(
            "breakpoint command list 2",
            error=True,
            startstr="error: '2' is not a currently valid breakpoint ID.")

        # The breakpoint list now only contains breakpoint 1.
        self.expect(
            "breakpoint list -f",
            "Breakpoint 1 exists",
            patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1, resolved = 1" %
                self.line,
                "hit count = 1"])

        # Not breakpoint 2.
        self.expect(
            "breakpoint list -f",
            "No more breakpoint 2",
            matching=False,
            substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1, resolved = 1" %
                self.line])

        # Run the program again, with breakpoint 1 remaining.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to breakpoint 1.

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1, since we reset counts
        # for each run.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

    def breakpoint_command_script_parameters(self):
        """Test that the frame and breakpoint location are being properly passed to the script breakpoint command function."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        # Now add callbacks for the breakpoints just created.
        self.runCmd("breakpoint command add -s python -o 'import side_effect; side_effect.frame = str(frame); side_effect.bp_loc = str(bp_loc)' 1")

        # Reset canary variables and run.
        side_effect.frame = None
        side_effect.bp_loc = None
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(side_effect.frame, exe=False, startstr="frame #0:")
        self.expect(side_effect.bp_loc, exe=False,
                patterns=["1.* where = .*main .* resolved,( hardware,)? hit count = 1"])

    def breakpoint_commands_on_creation(self):
        """Test that setting breakpoint commands when creating the breakpoint works"""
        target = self.createTestTarget()

        # Add a breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True,
            extra_options='-C bt -C "thread list" -C continue')

        bkpt = target.FindBreakpointByID(1)
        self.assertTrue(bkpt.IsValid(), "Couldn't find breakpoint 1")
        com_list = lldb.SBStringList()
        bkpt.GetCommandLineCommands(com_list)
        self.assertEqual(com_list.GetSize(), 3, "Got the wrong number of commands")
        self.assertEqual(com_list.GetStringAtIndex(0), "bt", "First bt")
        self.assertEqual(com_list.GetStringAtIndex(1), "thread list", "Next thread list")
        self.assertEqual(com_list.GetStringAtIndex(2), "continue", "Last continue")

    def test_add_commands_by_breakpoint_name(self):
        """Make sure that when you specify a breakpoint name to "break command add"
           it gets added to all the breakpoints marked with that name."""
        self.build()
        target = self.createTestTarget()

        bp_ids = []
        bp_names = ["main", "not_here", "main"]
        for bp_name in bp_names:
            bp = target.BreakpointCreateByName(bp_name)
            bp.AddName("MyBKPTS")
            bp_ids.append(bp.GetID())
        # First do it with a script one-liner:
        self.runCmd("breakpoint command add -s py -o 'print(\"some command\")' MyBKPTS")
        for id in bp_ids:
            self.expect("breakpoint command list {0}".format(id),
                        patterns=["some command"])
        # Now do the same thing with a python function:
        import side_effect
        self.runCmd("command script import --allow-reload ./bktptcmd.py")

        self.runCmd("breakpoint command add --python-function bktptcmd.function MyBKPTS")
        for id in bp_ids:
            self.expect("breakpoint command list {0}".format(id),
                        patterns=["bktptcmd.function"])



    def test_breakpoint_delete_disabled(self):
        """Test 'break delete --disabled' works"""
        self.build()
        target = self.createTestTarget()

        bp_1 = target.BreakpointCreateByName("main")
        bp_2 = target.BreakpointCreateByName("not_here")
        bp_3 = target.BreakpointCreateByName("main")
        bp_3.AddName("DeleteMeNot")

        bp_1.SetEnabled(False)
        bp_3.SetEnabled(False)

        bp_id_1 = bp_1.GetID()
        bp_id_2 = bp_2.GetID()
        bp_id_3 = bp_3.GetID()

        self.runCmd("breakpoint delete --disabled DeleteMeNot")

        bp_1 = target.FindBreakpointByID(bp_id_1)
        self.assertFalse(bp_1.IsValid(), "Didn't delete disabled breakpoint 1")

        bp_2 = target.FindBreakpointByID(bp_id_2)
        self.assertTrue(bp_2.IsValid(), "Deleted enabled breakpoint 2")

        bp_3 = target.FindBreakpointByID(bp_id_3)
        self.assertTrue(bp_3.IsValid(), "DeleteMeNot didn't protect disabled breakpoint 3")

        # Reset the first breakpoint, disable it, and do this again with no protected name:
        bp_1 = target.BreakpointCreateByName("main")

        bp_1.SetEnabled(False)

        bp_id_1 = bp_1.GetID()

        self.runCmd("breakpoint delete --disabled")

        bp_1 = target.FindBreakpointByID(bp_id_1)
        self.assertFalse(bp_1.IsValid(), "Didn't delete disabled breakpoint 1")

        bp_2 = target.FindBreakpointByID(bp_id_2)
        self.assertTrue(bp_2.IsValid(), "Deleted enabled breakpoint 2")

        bp_3 = target.FindBreakpointByID(bp_id_3)
        self.assertFalse(bp_3.IsValid(), "Didn't delete disabled breakpoint 3")
