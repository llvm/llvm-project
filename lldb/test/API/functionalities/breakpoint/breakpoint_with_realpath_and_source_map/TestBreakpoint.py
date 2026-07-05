"""
Test lldb breakpoint with symlinks/realpath and source-map.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil, lldbplatformutil


class BreakpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line_in_main = line_number("main.c", "// Set break point at this line.")
        self.line_in_foo = line_number("real/foo.h", "// Set break point at this line.")
        self.line_in_bar = line_number("real/bar.h", "// Set break point at this line.")
        self.line_in_qux = line_number("real/qux.h", "// Set break point at this line.")
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

    def buildAndCreateTarget(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

    @skipIf(oslist=["windows"])
    @skipIf(hostoslist=["windows"])
    def test_file_line_breakpoint_realpath_and_source_map(self):
        """Test file/line breakpoint with realpathing and source-mapping."""
        self.buildAndCreateTarget()
        cwd = os.getcwd()

        ######################################################################
        # Baseline
        # --------------------------------------------------------------------
        # Breakpoints should be resolved with paths which are in the line-table.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line_in_main, num_expected_locations=1, loc_exact=True
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "symlink1/foo.h",
            self.line_in_foo,
            num_expected_locations=1,
            loc_exact=True,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "symlink2/bar.h",
            self.line_in_bar,
            num_expected_locations=1,
            loc_exact=True,
        )
        lldbutil.run_break_set_by_file_and_line(
            self,
            "symlink2/qux.h",
            self.line_in_qux,
            num_expected_locations=1,
            loc_exact=True,
        )

        ######################################################################
        # Symlinked file
        # --------------------------------------------------------------------
        # - `symlink1/foo.h` is a symlink file, pointing at `real/foo.h`
        # - main.c includes `symlink1/foo.h`.
        # - As a result, the line-table contains a support file `(test_base_dir)/symlink1/foo.h`
        # - Setting a breakpoint for `real/foo.h` won't be resolved, because it doesn't match the above path.
        # - Setting a realpath prefix to the current working directory will cause the above support file to be realpath'ed to `(test_base_dir)/real/foo.h`
        # - Now setting a breakpoint for `real/foo.h` will be resolved.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "real/foo.h",
            self.line_in_foo,
            num_expected_locations=0,
            loc_exact=True,
        )
        self.runCmd(f'settings set target.source-realpath-prefixes "{cwd}"')
        lldbutil.run_break_set_by_file_and_line(
            self,
            "real/foo.h",
            self.line_in_foo,
            num_expected_locations=1,
            loc_exact=True,
        )
        # Clear settings so that the test below won't be affected.
        self.runCmd("settings clear target.source-realpath-prefixes")

        ######################################################################
        # Symlinked directory
        # --------------------------------------------------------------------
        # - `symlink2` is a symlink directory, pointing at `real`.
        # - So, `symlink2/bar.h` will be realpath'ed to `real/bar.h`.
        # - main.c includes `symlink2/bar.h`.
        # - As a result, the line-table contains a support file `(test_base_dir)/symlink2/bar.h`
        # - Setting a breakpoint for `real/bar.h` won't be resolved, because it doesn't match the above path.
        # - Setting a realpath prefix to the current working directory will cause the above support file to be realpath'ed to `(test_base_dir)/real/bar.h`
        # - Now setting a breakpoint for `real/bar.h` will be resolved.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "real/bar.h",
            self.line_in_foo,
            num_expected_locations=0,
            loc_exact=True,
        )
        self.runCmd(f'settings set target.source-realpath-prefixes "{cwd}"')
        lldbutil.run_break_set_by_file_and_line(
            self,
            "real/bar.h",
            self.line_in_foo,
            num_expected_locations=1,
            loc_exact=True,
        )
        # Clear settings so that the test below won't be affected.
        self.runCmd("settings clear target.source-realpath-prefixes")

        ######################################################################
        # Symlink + source-map
        # --------------------------------------------------------------------
        # - `symlink2` is a symlink directory, pointing at `real`.
        # - So, `symlink2/qux.h` will be realpath'ed to `real/qux.h`.
        # - main.c includes `symlink2/qux.h`.
        # - As a result, the line-table contains a support file `(test_base_dir)/symlink2/qux.h`
        # - Setting a realpath prefix to the current working directory will cause the above support file to be realpath'ed to `(test_base_dir)/real/qux.h`
        # - Setting a breakpoint for `to-be-mapped/qux.h` won't be resolved, because it doesn't match the above path.
        # - After setting a source-map, setting the same breakpoint will be resolved, because the input path `to-be-mapped/qux.h` is reverse-mapped to `real/qux.h`, which matches the realpath'ed support file.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "real/qux.h",
            self.line_in_foo,
            num_expected_locations=0,
            loc_exact=True,
        )
        self.runCmd(f'settings set target.source-realpath-prefixes "{cwd}"')
        lldbutil.run_break_set_by_file_and_line(
            self,
            "to-be-mapped/qux.h",
            self.line_in_foo,
            num_expected_locations=0,
            loc_exact=True,
        )
        self.runCmd('settings set target.source-map "real" "to-be-mapped"')
        lldbutil.run_break_set_by_file_and_line(
            self,
            "to-be-mapped/qux.h",
            self.line_in_foo,
            num_expected_locations=1,
            loc_exact=True,
        )
        # Clear settings so that the test below won't be affected.
        self.runCmd("settings clear target.source-realpath-prefixes")
