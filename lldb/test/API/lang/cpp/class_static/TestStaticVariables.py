"""
Test display and Python APIs on file and class static variables.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StaticVariableTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def test_with_run_command(self):
        """Test that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Global variables are no longer displayed with the "frame variable"
        # command.
        self.expect(
            "target variable A::g_points",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\(PointType\[[1-9]*\]\) A::g_points = {"],
        )
        self.expect(
            "target variable g_points",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(PointType[2]) g_points"],
        )

        # On Mac OS X, gcc 4.2 emits the wrong debug info for A::g_points.
        # A::g_points is an array of two elements.
        if self.platformIsDarwin() or self.getPlatform() == "linux":
            self.expect(
                "target variable A::g_points[1].x",
                VARIABLES_DISPLAYED_CORRECTLY,
                startstr="(int) A::g_points[1].x = 11",
            )

    @expectedFailureAll(
        compiler=["gcc"], bugnumber="Compiler emits incomplete debug info"
    )
    @expectedFailureAll(
        compiler=["clang"], compiler_version=["<", "3.9"], bugnumber="llvm.org/pr20550"
    )
    def test_with_run_command_complete(self):
        """
        Test that file and class static variables display correctly with
        complete debug information.
        """
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Global variables are no longer displayed with the "frame variable"
        # command.
        self.expect(
            "target variable A::g_points",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[
                "\(PointType\[[1-9]*\]\) A::g_points = {",
                "(x = 1, y = 2)",
                "(x = 11, y = 22)",
            ],
        )

        # Ensure that we take the context into account and only print
        # A::g_points.
        self.expect(
            "target variable A::g_points",
            VARIABLES_DISPLAYED_CORRECTLY,
            matching=False,
            patterns=["(x = 3, y = 4)", "(x = 33, y = 44)"],
        )

        # Finally, ensure that we print both points when not specifying a
        # context.
        self.expect(
            "target variable g_points",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(PointType[2]) g_points",
                "(x = 1, y = 2)",
                "(x = 11, y = 22)",
                "(x = 3, y = 4)",
                "(x = 33, y = 44)",
            ],
        )

    def build_value_check(self, var_name, values):
        children_1 = [
            ValueCheck(name="x", value=values[0], type="int"),
            ValueCheck(name="y", value=values[1], type="int"),
        ]
        children_2 = [
            ValueCheck(name="x", value=values[2], type="int"),
            ValueCheck(name="y", value=values[3], type="int"),
        ]
        elem_0 = ValueCheck(
            name="[0]", value=None, type="PointType", children=children_1
        )
        elem_1 = ValueCheck(
            name="[1]", value=None, type="PointType", children=children_2
        )
        value_check = ValueCheck(
            name=var_name, value=None, type="PointType[2]", children=[elem_0, elem_1]
        )

        return value_check

    @expectedFailureAll(
        compiler=["gcc"], bugnumber="Compiler emits incomplete debug info"
    )
    @expectedFailureAll(
        compiler=["clang"], compiler_version=["<", "3.9"], bugnumber="llvm.org/pr20550"
    )
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    @add_test_categories(["pyapi"])
    def test_with_python_FindValue(self):
        """Test Python APIs on file and class static variables."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Get the SBValue of 'A::g_points' and 'g_points'.
        frame = thread.GetFrameAtIndex(0)

        # arguments =>     False
        # locals =>        False
        # statics =>       True
        # in_scope_only => False
        valList = frame.GetVariables(False, False, True, False)

        # Build ValueCheckers for the values we're going to find:
        value_check_A = self.build_value_check("A::g_points", ["1", "2", "11", "22"])
        value_check_none = self.build_value_check("g_points", ["3", "4", "33", "44"])
        value_check_AA = self.build_value_check("AA::g_points", ["5", "6", "55", "66"])

        for val in valList:
            self.DebugSBValue(val)
            name = val.GetName()
            self.assertIn(name, ["g_points", "A::g_points", "AA::g_points"])

            if name == "A::g_points":
                self.assertEqual(val.GetValueType(), lldb.eValueTypeVariableGlobal)
                value_check_A.check_value(self, val, "Got A::g_points right")
            if name == "g_points":
                self.assertEqual(val.GetValueType(), lldb.eValueTypeVariableStatic)
                value_check_none.check_value(self, val, "Got g_points right")
            if name == "AA::g_points":
                self.assertEqual(val.GetValueType(), lldb.eValueTypeVariableGlobal)
                value_check_AA.check_value(self, val, "Got AA::g_points right")

        # SBFrame.FindValue() should also work.
        val = frame.FindValue("A::g_points", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)
        value_check_A.check_value(self, val, "FindValue also works")

        # Also exercise the "parameter" and "local" scopes while we are at it.
        val = frame.FindValue("argc", lldb.eValueTypeVariableArgument)
        self.DebugSBValue(val)
        self.assertEqual(val.GetName(), "argc")

        val = frame.FindValue("argv", lldb.eValueTypeVariableArgument)
        self.DebugSBValue(val)
        self.assertEqual(val.GetName(), "argv")

        val = frame.FindValue("hello_world", lldb.eValueTypeVariableLocal)
        self.DebugSBValue(val)
        self.assertEqual(val.GetName(), "hello_world")

    # This test tests behavior that's been broken for a very long time..
    # The fix for it is in the accelerator table part of the DWARF reader,
    # and I fixed the version that the names accelerator uses, but I don't
    # know how to fix it on systems that don't use that. There isn't a
    # decorator for that - not sure how to construct that so I'm limiting the
    # test do Darwin for now.
    @expectedFailureAll(
        compiler=["gcc"], bugnumber="Compiler emits incomplete debug info"
    )
    @skipUnlessDarwin
    @add_test_categories(["pyapi"])
    def test_with_python_FindGlobalVariables(self):
        """Test Python APIs on file and class static variables."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Get the SBValue of 'A::g_points' and 'g_points'.
        frame = thread.GetFrameAtIndex(0)

        # Build ValueCheckers for the values we're going to find:
        value_check_A = self.build_value_check("A::g_points", ["1", "2", "11", "22"])
        value_check_none = self.build_value_check("g_points", ["3", "4", "33", "44"])
        value_check_AA = self.build_value_check("AA::g_points", ["5", "6", "55", "66"])

        # We should also be able to get class statics from FindGlobalVariables.
        # eMatchTypeStartsWith should only find A:: not AA::
        val_list = target.FindGlobalVariables("A::", 10, lldb.eMatchTypeStartsWith)
        self.assertEqual(val_list.GetSize(), 1, "Found only one match")
        val = val_list[0]
        value_check_A.check_value(self, val, "FindGlobalVariables starts with")

        # Regex should find both
        val_list = target.FindGlobalVariables("A::", 10, lldb.eMatchTypeRegex)
        self.assertEqual(val_list.GetSize(), 2, "Found A & AA")
        found_a = False
        found_aa = False
        for val in val_list:
            name = val.GetName()
            if name == "A::g_points":
                value_check_A.check_value(self, val, "AA found by regex")
                found_a = True
            elif name == "AA::g_points":
                value_check_AA.check_value(self, val, "A found by regex")
                found_aa = True

        self.assertTrue(found_a, "Regex search found A::g_points")
        self.assertTrue(found_aa, "Regex search found AA::g_points")

        # Regex lowercase should find both as well.
        val_list = target.FindGlobalVariables(
            "a::g_points", 10, lldb.eMatchTypeRegexInsensitive
        )
        self.assertEqual(val_list.GetSize(), 2, "Found A & AA")

        # Normal search for full name should find one, but it looks like we don't match
        # on identifier boundaries here yet:
        val_list = target.FindGlobalVariables("A::g_points", 10, lldb.eMatchTypeNormal)
        self.assertEqual(
            val_list.GetSize(), 2, "We aren't matching on name boundaries yet"
        )

        # Normal search for g_points should find 3 - FindGlobalVariables doesn't distinguish
        # between file statics and globals:
        val_list = target.FindGlobalVariables("g_points", 10, lldb.eMatchTypeNormal)
        self.assertEqual(val_list.GetSize(), 3, "Found all three g_points")
