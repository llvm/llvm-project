"""Test that hidden ivars in a shared library are visible from the main executable."""


import subprocess

import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class HiddenIvarsTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "main.m"
        self.line = line_number(self.source, "// breakpoint1")
        # The makefile names of the shared libraries as they appear in DYLIB_NAME.
        # The names should have no loading "lib" or extension as they will be
        # localized
        self.shlib_names = ["InternalDefiner"]

    @skipIf(
        debug_info=no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM",
    )
    def test_expr_stripped(self):
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")
        else:
            self.build()
            self.expr(True)

    def test_expr(self):
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")
        else:
            self.build()
            self.expr(False)

    @skipIf(
        debug_info=no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM",
    )
    def test_frame_variable_stripped(self):
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")
        else:
            self.build()
            self.frame_var(True)

    def test_frame_variable(self):
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")
        else:
            self.build()
            self.frame_var(False)

    @unittest2.expectedFailure  # rdar://18683637
    def test_frame_variable_across_modules(self):
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")
        else:
            self.build()
            self.common_setup(False)
            self.expect(
                "frame variable k->bar", VARIABLES_DISPLAYED_CORRECTLY, substrs=["= 3"]
            )

    def common_setup(self, strip):
        if strip:
            exe = self.getBuildArtifact("stripped/a.out")
        else:
            exe = self.getBuildArtifact("a.out")
        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(target, self.shlib_names)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory()
        )
        self.assertTrue(process, PROCESS_IS_VALID)

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

    def expr(self, strip):
        self.common_setup(strip)

        # This should display correctly.
        self.expect(
            "expression (j->_definer->foo)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["= 4"],
        )

        self.expect(
            "expression (j->_definer->bar)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["= 5"],
        )

        if strip:
            self.expect(
                "expression *(j->_definer)",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 4"],
            )
        else:
            self.expect(
                "expression *(j->_definer)",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 4", "bar = 5"],
            )

        self.expect(
            "expression (k->foo)", VARIABLES_DISPLAYED_CORRECTLY, substrs=["= 2"]
        )

        self.expect(
            "expression (k->bar)", VARIABLES_DISPLAYED_CORRECTLY, substrs=["= 3"]
        )

        self.expect(
            "expression k.filteredDataSource",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[" = 0x", '"2 elements"'],
        )

        if strip:
            self.expect(
                "expression *(k)",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 2", " = 0x", '"2 elements"'],
            )
        else:
            self.expect(
                "expression *(k)",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=[
                    "foo = 2",
                    "bar = 3",
                    "_filteredDataSource = 0x",
                    '"2 elements"',
                ],
            )

    def frame_var(self, strip):
        self.common_setup(strip)

        # This should display correctly.
        self.expect(
            "frame variable j->_definer->foo",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["= 4"],
        )

        if not strip:
            self.expect(
                "frame variable j->_definer->bar",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["= 5"],
            )

        if strip:
            self.expect(
                "frame variable *j->_definer",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 4"],
            )
        else:
            self.expect(
                "frame variable *j->_definer",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 4", "bar = 5"],
            )

        self.expect(
            "frame variable k->foo", VARIABLES_DISPLAYED_CORRECTLY, substrs=["= 2"]
        )

        self.expect(
            "frame variable k->_filteredDataSource",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[" = 0x", '"2 elements"'],
        )

        if strip:
            self.expect(
                "frame variable *k",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=["foo = 2", "_filteredDataSource = 0x", '"2 elements"'],
            )
        else:
            self.expect(
                "frame variable *k",
                VARIABLES_DISPLAYED_CORRECTLY,
                substrs=[
                    "foo = 2",
                    "bar = 3",
                    "_filteredDataSource = 0x",
                    '"2 elements"',
                ],
            )
