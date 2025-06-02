"""
Test lldb-dap setBreakpoints request
"""


from dap_server import Source
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


@skip("Temporarily disable the breakpoint tests")
class TestDAP_setBreakpoints(lldbdap_testcase.DAPTestCaseBase):
    def setUp(self):
        lldbdap_testcase.DAPTestCaseBase.setUp(self)

        self.main_basename = "main-copy.cpp"
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    @skipIfWindows
    def test_source_map(self):
        """
        This test simulates building two files in a folder, and then moving
        each source to a different folder. Then, the debug session is started
        with the corresponding source maps to have breakpoints and frames
        working.
        """
        self.build_and_create_debug_adapter()

        other_basename = "other-copy.c"
        other_path = self.getBuildArtifact(other_basename)

        source_folder = os.path.dirname(self.main_path)

        new_main_folder = os.path.join(source_folder, "moved_main")
        new_other_folder = os.path.join(source_folder, "moved_other")

        new_main_path = os.path.join(new_main_folder, self.main_basename)
        new_other_path = os.path.join(new_other_folder, other_basename)

        # move the sources
        os.mkdir(new_main_folder)
        os.mkdir(new_other_folder)
        shutil.move(self.main_path, new_main_path)
        shutil.move(other_path, new_other_path)

        main_line = line_number("main.cpp", "break 12")
        other_line = line_number("other.c", "break other")

        program = self.getBuildArtifact("a.out")
        source_map = [
            [source_folder, new_main_folder],
            [source_folder, new_other_folder],
        ]
        self.launch(program, sourceMap=source_map)

        # breakpoint in main.cpp
        response = self.dap_server.request_setBreakpoints(
            Source(new_main_path), [main_line]
        )
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 1)
        breakpoint = breakpoints[0]
        self.assertEqual(breakpoint["line"], main_line)
        self.assertTrue(breakpoint["verified"])
        self.assertEqual(self.main_basename, breakpoint["source"]["name"])
        self.assertEqual(new_main_path, breakpoint["source"]["path"])

        # 2nd breakpoint, which is from a dynamically loaded library
        response = self.dap_server.request_setBreakpoints(
            Source(new_other_path), [other_line]
        )
        breakpoints = response["body"]["breakpoints"]
        breakpoint = breakpoints[0]
        self.assertEqual(breakpoint["line"], other_line)
        self.assertFalse(breakpoint["verified"])
        self.assertEqual(other_basename, breakpoint["source"]["name"])
        self.assertEqual(new_other_path, breakpoint["source"]["path"])
        other_breakpoint_id = breakpoint["id"]

        self.dap_server.request_continue()
        self.verify_breakpoint_hit([other_breakpoint_id])

        # 2nd breakpoint again, which should be valid at this point
        response = self.dap_server.request_setBreakpoints(
            Source(new_other_path), [other_line]
        )
        breakpoints = response["body"]["breakpoints"]
        breakpoint = breakpoints[0]
        self.assertEqual(breakpoint["line"], other_line)
        self.assertTrue(breakpoint["verified"])
        self.assertEqual(other_basename, breakpoint["source"]["name"])
        self.assertEqual(new_other_path, breakpoint["source"]["path"])

        # now we check the stack trace making sure that we got mapped source paths
        frames = self.dap_server.request_stackTrace()["body"]["stackFrames"]

        self.assertEqual(frames[0]["source"]["name"], other_basename)
        self.assertEqual(frames[0]["source"]["path"], new_other_path)

        self.assertEqual(frames[1]["source"]["name"], self.main_basename)
        self.assertEqual(frames[1]["source"]["path"], new_main_path)

    @skipIfWindows
    def test_set_and_clear(self):
        """Tests setting and clearing source file and line breakpoints.
        This packet is a bit tricky on the debug adapter side since there
        is no "clearBreakpoints" packet. Source file and line breakpoints
        are set by sending a "setBreakpoints" packet with a source file
        specified and zero or more source lines. If breakpoints have been
        set in the source file before, any existing breakpoints must remain
        set, and any new breakpoints must be created, and any breakpoints
        that were in previous requests and are not in the current request
        must be removed. This function tests this setting and clearing
        and makes sure things happen correctly. It doesn't test hitting
        breakpoints and the functionality of each breakpoint, like
        'conditions' and 'hitCondition' settings."""
        first_line = line_number("main.cpp", "break 12")
        second_line = line_number("main.cpp", "break 13")
        third_line = line_number("main.cpp", "break 14")
        lines = [first_line, third_line, second_line]

        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set 3 breakpoints and verify that they got set correctly
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        line_to_id = {}
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints),
            len(lines),
            "expect %u source breakpoints" % (len(lines)),
        )
        for index, breakpoint in enumerate(breakpoints):
            line = breakpoint["line"]
            self.assertEqual(line, lines[index])
            # Store the "id" of the breakpoint that was set for later
            line_to_id[line] = breakpoint["id"]
            self.assertTrue(breakpoint["verified"], "expect breakpoint verified")

        # There is no breakpoint delete packet, clients just send another
        # setBreakpoints packet with the same source file with fewer lines.
        # Below we remove the second line entry and call the setBreakpoints
        # function again. We want to verify that any breakpoints that were set
        # before still have the same "id". This means we didn't clear the
        # breakpoint and set it again at the same location. We also need to
        # verify that the second line location was actually removed.
        lines.remove(second_line)
        # Set 2 breakpoints and verify that the previous breakpoints that were
        # set above are still set.
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints),
            len(lines),
            "expect %u source breakpoints" % (len(lines)),
        )
        for index, breakpoint in enumerate(breakpoints):
            line = breakpoint["line"]
            self.assertEqual(line, lines[index])
            # Verify the same breakpoints are still set within LLDB by
            # making sure the breakpoint ID didn't change
            self.assertEqual(
                line_to_id[line],
                breakpoint["id"],
                "verify previous breakpoints stayed the same",
            )
            self.assertTrue(breakpoint["verified"], "expect breakpoint still verified")

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target
        response = self.dap_server.request_testGetTargetBreakpoints()
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints),
            len(lines),
            "expect %u source breakpoints" % (len(lines)),
        )
        for breakpoint in breakpoints:
            line = breakpoint["line"]
            # Verify the same breakpoints are still set within LLDB by
            # making sure the breakpoint ID didn't change
            self.assertEqual(
                line_to_id[line],
                breakpoint["id"],
                "verify previous breakpoints stayed the same",
            )
            self.assertIn(line, lines, "line expected in lines array")
            self.assertTrue(breakpoint["verified"], "expect breakpoint still verified")

        # Now clear all breakpoints for the source file by passing down an
        # empty lines array
        lines = []
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints),
            len(lines),
            "expect %u source breakpoints" % (len(lines)),
        )

        # Verify with the target that all breakpoints have been cleared
        response = self.dap_server.request_testGetTargetBreakpoints()
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints),
            len(lines),
            "expect %u source breakpoints" % (len(lines)),
        )

        # Now set a breakpoint again in the same source file and verify it
        # was added.
        lines = [second_line]
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        if response:
            breakpoints = response["body"]["breakpoints"]
            self.assertEqual(
                len(breakpoints),
                len(lines),
                "expect %u source breakpoints" % (len(lines)),
            )
            for breakpoint in breakpoints:
                line = breakpoint["line"]
                self.assertIn(line, lines, "line expected in lines array")
                self.assertTrue(
                    breakpoint["verified"], "expect breakpoint still verified"
                )

        # Now get the full list of breakpoints set in the target and verify
        # we have only 2 breakpoints set. The response above could have told
        # us about 2 breakpoints, but we want to make sure we don't have the
        # third one still set in the target
        response = self.dap_server.request_testGetTargetBreakpoints()
        if response:
            breakpoints = response["body"]["breakpoints"]
            self.assertEqual(
                len(breakpoints),
                len(lines),
                "expect %u source breakpoints" % (len(lines)),
            )
            for breakpoint in breakpoints:
                line = breakpoint["line"]
                self.assertIn(line, lines, "line expected in lines array")
                self.assertTrue(
                    breakpoint["verified"], "expect breakpoint still verified"
                )

    @skipIfWindows
    def test_clear_breakpoints_unset_breakpoints(self):
        """Test clearing breakpoints like test_set_and_clear, but clear
        breakpoints by omitting the breakpoints array instead of sending an
        empty one."""
        lines = [
            line_number("main.cpp", "break 12"),
            line_number("main.cpp", "break 13"),
        ]

        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set one breakpoint and verify that it got set correctly.
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        line_to_id = {}
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(
            len(breakpoints), len(lines), "expect %u source breakpoints" % (len(lines))
        )
        for index, breakpoint in enumerate(breakpoints):
            line = breakpoint["line"]
            self.assertEqual(line, lines[index])
            # Store the "id" of the breakpoint that was set for later
            line_to_id[line] = breakpoint["id"]
            self.assertTrue(breakpoint["verified"], "expect breakpoint verified")

        # Now clear all breakpoints for the source file by not setting the
        # lines array.
        lines = None
        response = self.dap_server.request_setBreakpoints(Source(self.main_path), lines)
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 0, "expect no source breakpoints")

        # Verify with the target that all breakpoints have been cleared.
        response = self.dap_server.request_testGetTargetBreakpoints()
        breakpoints = response["body"]["breakpoints"]
        self.assertEqual(len(breakpoints), 0, "expect no source breakpoints")

    @skipIfWindows
    def test_functionality(self):
        """Tests hitting breakpoints and the functionality of a single
        breakpoint, like 'conditions' and 'hitCondition' settings."""
        loop_line = line_number("main.cpp", "// break loop")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        # Set a breakpoint at the loop line with no condition and no
        # hitCondition
        breakpoint_ids = self.set_source_breakpoints(self.main_path, [loop_line])
        self.assertEqual(len(breakpoint_ids), 1, "expect one breakpoint")
        self.dap_server.request_continue()

        # Verify we hit the breakpoint we just set
        self.verify_breakpoint_hit(breakpoint_ids)

        # Make sure i is zero at first breakpoint
        i = int(self.dap_server.get_local_variable_value("i"))
        self.assertEqual(i, 0, "i != 0 after hitting breakpoint")

        # Update the condition on our breakpoint
        new_breakpoint_ids = self.set_source_breakpoints(
            self.main_path, [loop_line], [{"condition": "i==4"}]
        )
        self.assertEqual(
            breakpoint_ids,
            new_breakpoint_ids,
            "existing breakpoint should have its condition " "updated",
        )

        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.dap_server.get_local_variable_value("i"))
        self.assertEqual(i, 4, "i != 4 showing conditional works")

        new_breakpoint_ids = self.set_source_breakpoints(
            self.main_path, [loop_line], [{"hitCondition": "2"}]
        )

        self.assertEqual(
            breakpoint_ids,
            new_breakpoint_ids,
            "existing breakpoint should have its condition " "updated",
        )

        # Continue with a hitCondition of 2 and expect it to skip 1 value
        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.dap_server.get_local_variable_value("i"))
        self.assertEqual(i, 6, "i != 6 showing hitCondition works")

        # continue after hitting our hitCondition and make sure it only goes
        # up by 1
        self.continue_to_breakpoints(breakpoint_ids)
        i = int(self.dap_server.get_local_variable_value("i"))
        self.assertEqual(i, 7, "i != 7 showing post hitCondition hits every time")

    @skipIfWindows
    def test_column_breakpoints(self):
        """Test setting multiple breakpoints in the same line at different columns."""
        loop_line = line_number("main.cpp", "// break loop")

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set two breakpoints on the loop line at different columns.
        columns = [13, 39]
        response = self.dap_server.request_setBreakpoints(
            Source(self.main_path),
            [loop_line, loop_line],
            list({"column": c} for c in columns),
        )

        # Verify the breakpoints were set correctly
        breakpoints = response["body"]["breakpoints"]
        breakpoint_ids = []
        self.assertEqual(
            len(breakpoints),
            len(columns),
            "expect %u source breakpoints" % (len(columns)),
        )
        for index, breakpoint in enumerate(breakpoints):
            self.assertEqual(breakpoint["line"], loop_line)
            self.assertEqual(breakpoint["column"], columns[index])
            self.assertTrue(breakpoint["verified"], "expect breakpoint verified")
            breakpoint_ids.append(breakpoint["id"])

        # Continue to the first breakpoint,
        self.continue_to_breakpoints([breakpoint_ids[0]])

        # We should have stopped right before the call to `twelve`.
        # Step into and check we are inside `twelve`.
        self.stepIn()
        func_name = self.get_stackFrames()[0]["name"]
        self.assertEqual(func_name, "twelve(int)")

        # Continue to the second breakpoint.
        self.continue_to_breakpoints([breakpoint_ids[1]])

        # We should have stopped right before the call to `fourteen`.
        # Step into and check we are inside `fourteen`.
        self.stepIn()
        func_name = self.get_stackFrames()[0]["name"]
        self.assertEqual(func_name, "a::fourteen(int)")
