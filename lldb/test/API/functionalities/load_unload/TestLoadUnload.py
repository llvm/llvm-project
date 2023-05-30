"""
Test that breakpoint by symbol name works correctly with dynamic libs.
"""

import os
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LoadUnloadTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.setup_test()
        # Invoke the default build rule.
        self.build()
        # Find the line number to break for main.cpp.
        self.line = line_number(
            "main.cpp",
            "// Set break point at this line for test_lldb_process_load_and_unload_commands().",
        )
        self.line_d_function = line_number(
            "d.cpp", "// Find this line number within d_dunction()."
        )

    def setup_test(self):
        lldbutil.mkdir_p(self.getBuildArtifact("hidden"))
        if lldb.remote_platform:
            path = lldb.remote_platform.GetWorkingDirectory()
        else:
            path = self.getBuildDir()
            if self.dylibPath in os.environ:
                sep = self.platformContext.shlib_path_separator
                path = os.environ[self.dylibPath] + sep + path
        self.runCmd(
            "settings append target.env-vars '{}={}'".format(self.dylibPath, path)
        )
        self.default_path = path

    def copy_shlibs_to_remote(self, hidden_dir=False):
        """Copies the shared libs required by this test suite to remote.
        Does nothing in case of non-remote platforms.
        """
        if lldb.remote_platform:
            ext = "so"
            if self.platformIsDarwin():
                ext = "dylib"

            shlibs = [
                "libloadunload_a." + ext,
                "libloadunload_b." + ext,
                "libloadunload_c." + ext,
                "libloadunload_d." + ext,
            ]
            wd = lldb.remote_platform.GetWorkingDirectory()
            cwd = os.getcwd()
            for f in shlibs:
                err = lldb.remote_platform.Put(
                    lldb.SBFileSpec(self.getBuildArtifact(f)),
                    lldb.SBFileSpec(os.path.join(wd, f)),
                )
                if err.Fail():
                    raise RuntimeError(
                        "Unable copy '%s' to '%s'.\n>>> %s" % (f, wd, err.GetCString())
                    )
            if hidden_dir:
                shlib = "libloadunload_d." + ext
                hidden_dir = os.path.join(wd, "hidden")
                hidden_file = os.path.join(hidden_dir, shlib)
                err = lldb.remote_platform.MakeDirectory(hidden_dir)
                if err.Fail():
                    raise RuntimeError(
                        "Unable to create a directory '%s'." % hidden_dir
                    )
                err = lldb.remote_platform.Put(
                    lldb.SBFileSpec(os.path.join("hidden", shlib)),
                    lldb.SBFileSpec(hidden_file),
                )
                if err.Fail():
                    raise RuntimeError(
                        "Unable copy 'libloadunload_d.so' to '%s'.\n>>> %s"
                        % (wd, err.GetCString())
                    )

    def setSvr4Support(self, enabled):
        self.runCmd(
            "settings set plugin.process.gdb-remote.use-libraries-svr4 {enabled}".format(
                enabled="true" if enabled else "false"
            )
        )

    # libloadunload_d.so does not appear in the image list because executable
    # dependencies are resolved relative to the debuggers PWD. Bug?
    @expectedFailureAll(oslist=["freebsd", "linux", "netbsd"])
    @skipIfRemote
    @skipIfWindows  # Windows doesn't have dlopen and friends, dynamic libraries work differently
    def test_modules_search_paths(self):
        """Test target modules list after loading a different copy of the library libd.dylib, and verifies that it works with 'target modules search-paths add'."""
        if self.platformIsDarwin():
            dylibName = "libloadunload_d.dylib"
        else:
            dylibName = "libloadunload_d.so"

        # The directory with the dynamic library we did not link to.
        new_dir = os.path.join(self.getBuildDir(), "hidden")

        old_dylib = os.path.join(self.getBuildDir(), dylibName)
        new_dylib = os.path.join(new_dir, dylibName)
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("target modules list", substrs=[old_dylib])
        # self.expect("target modules list -t 3",
        #    patterns = ["%s-[^-]*-[^-]*" % self.getArchitecture()])
        # Add an image search path substitution pair.
        self.runCmd(
            "target modules search-paths add %s %s" % (self.getBuildDir(), new_dir)
        )

        self.expect(
            "target modules search-paths list", substrs=[self.getBuildDir(), new_dir]
        )

        self.expect(
            "target modules search-paths query %s" % self.getBuildDir(),
            "Image search path successfully transformed",
            substrs=[new_dir],
        )

        # Obliterate traces of libd from the old location.
        os.remove(old_dylib)
        # Inform (DY)LD_LIBRARY_PATH of the new path, too.
        env_cmd_string = (
            "settings replace target.env-vars " + self.dylibPath + "=" + new_dir
        )
        if self.TraceOn():
            print("Set environment to: ", env_cmd_string)
        self.runCmd(env_cmd_string)
        self.runCmd("settings show target.env-vars")

        self.runCmd("run")

        self.expect(
            "target modules list",
            "LLDB successfully locates the relocated dynamic library",
            substrs=[new_dylib],
        )

    # libloadunload_d.so does not appear in the image list because executable
    # dependencies are resolved relative to the debuggers PWD. Bug?
    @expectedFailureAll(oslist=["freebsd", "linux", "netbsd"])
    @expectedFailureAndroid  # wrong source file shows up for hidden library
    @skipIfWindows  # Windows doesn't have dlopen and friends, dynamic libraries work differently
    @skipIfDarwinEmbedded
    def test_dyld_library_path(self):
        """Test (DY)LD_LIBRARY_PATH after moving libd.dylib, which defines d_function, somewhere else."""
        self.copy_shlibs_to_remote(hidden_dir=True)

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Shut off ANSI color usage so we don't get ANSI escape sequences
        # mixed in with stop locations.
        self.dbg.SetUseColor(False)

        if self.platformIsDarwin():
            dylibName = "libloadunload_d.dylib"
            dsymName = "libloadunload_d.dylib.dSYM"
        else:
            dylibName = "libloadunload_d.so"

        # The directory to relocate the dynamic library and its debugging info.
        special_dir = "hidden"
        if lldb.remote_platform:
            wd = lldb.remote_platform.GetWorkingDirectory()
        else:
            wd = self.getBuildDir()

        old_dir = wd
        new_dir = os.path.join(wd, special_dir)
        old_dylib = os.path.join(old_dir, dylibName)

        # For now we don't track (DY)LD_LIBRARY_PATH, so the old
        # library will be in the modules list.
        self.expect(
            "target modules list", substrs=[os.path.basename(old_dylib)], matching=True
        )

        lldbutil.run_break_set_by_file_and_line(
            self, "d.cpp", self.line_d_function, num_expected_locations=1
        )
        # After run, make sure the non-hidden library is picked up.
        self.expect("run", substrs=["return", "700"])

        self.runCmd("continue")

        # Add the hidden directory first in the search path.
        env_cmd_string = "settings set target.env-vars %s=%s%s%s" % (
            self.dylibPath,
            new_dir,
            self.platformContext.shlib_path_separator,
            self.default_path,
        )
        self.runCmd(env_cmd_string)

        # This time, the hidden library should be picked up.
        self.expect("run", substrs=["return", "12345"])

    @expectedFailureAll(
        bugnumber="llvm.org/pr25805", hostoslist=["windows"], triple=".*-android"
    )
    @expectedFailureAll(oslist=["windows"])  # process load not implemented
    def test_lldb_process_load_and_unload_commands(self):
        self.setSvr4Support(False)
        self.run_lldb_process_load_and_unload_commands()

    @expectedFailureAll(
        bugnumber="llvm.org/pr25805", hostoslist=["windows"], triple=".*-android"
    )
    @expectedFailureAll(oslist=["windows"])  # process load not implemented
    def test_lldb_process_load_and_unload_commands_with_svr4(self):
        self.setSvr4Support(True)
        self.run_lldb_process_load_and_unload_commands()

    def run_lldb_process_load_and_unload_commands(self):
        """Test that lldb process load/unload command work correctly."""
        self.copy_shlibs_to_remote()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break at main.cpp before the call to dlopen().
        # Use lldb's process load command to load the dylib, instead.

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + "loadunload_a." + ctx.shlib_extension
        localDylibPath = self.getBuildArtifact(dylibName)
        if lldb.remote_platform:
            wd = lldb.remote_platform.GetWorkingDirectory()
            remoteDylibPath = lldbutil.join_remote_paths(wd, dylibName)
        else:
            remoteDylibPath = localDylibPath

        # First make sure that we get some kind of error if process load fails.
        # We print some error even if the load fails, which isn't formalized.
        # The only plugin at present (Posix) that supports this says "unknown reasons".
        # If another plugin shows up, let's require it uses "unknown error" as well.
        non_existant_shlib = "/NoSuchDir/NoSuchSubdir/ReallyNo/NotAFile"
        self.expect(
            "process load %s" % (non_existant_shlib),
            error=True,
            matching=False,
            patterns=["unknown reasons"],
        )

        # Make sure that a_function does not exist at this point.
        self.expect(
            "image lookup -n a_function",
            "a_function should not exist yet",
            error=True,
            matching=False,
            patterns=["1 match found"],
        )

        # Use lldb 'process load' to load the dylib.
        self.expect(
            "process load %s --install=%s" % (localDylibPath, remoteDylibPath),
            "%s loaded correctly" % dylibName,
            patterns=[
                'Loading "%s".*ok' % re.escape(localDylibPath),
                "Image [0-9]+ loaded",
            ],
        )

        # Search for and match the "Image ([0-9]+) loaded" pattern.
        output = self.res.GetOutput()
        pattern = re.compile("Image ([0-9]+) loaded")
        for l in output.split(os.linesep):
            self.trace("l:", l)
            match = pattern.search(l)
            if match:
                break
        index = match.group(1)

        # Now we should have an entry for a_function.
        self.expect(
            "image lookup -n a_function",
            "a_function should now exist",
            patterns=["1 match found .*%s" % dylibName],
        )

        # Use lldb 'process unload' to unload the dylib.
        self.expect(
            "process unload %s" % index,
            "%s unloaded correctly" % dylibName,
            patterns=["Unloading .* with index %s.*ok" % index],
        )

        self.runCmd("process continue")

    @expectedFailureAll(oslist=["windows"])  # breakpoint not hit
    def test_load_unload(self):
        self.setSvr4Support(False)
        self.run_load_unload()

    @expectedFailureAll(oslist=["windows"])  # breakpoint not hit
    def test_load_unload_with_svr4(self):
        self.setSvr4Support(True)
        self.run_load_unload()

    def run_load_unload(self):
        """Test breakpoint by name works correctly with dlopen'ing."""
        self.copy_shlibs_to_remote()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        lldbutil.run_break_set_by_symbol(self, "a_function", num_expected_locations=0)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "a_function", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        # Issue the 'continue' command.  We should stop again at a_function.
        # The stop reason of the thread should be breakpoint and at a_function.
        self.runCmd("continue")

        # rdar://problem/8508987
        # The a_function breakpoint should be encountered twice.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "a_function", "stop reason = breakpoint"],
        )

        # The breakpoint should have a hit count of 2.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=2)

    def test_step_over_load(self):
        self.setSvr4Support(False)
        self.run_step_over_load()

    def test_step_over_load_with_svr4(self):
        self.setSvr4Support(True)
        self.run_step_over_load()

    def run_step_over_load(self):
        """Test stepping over code that loads a shared library works correctly."""
        self.copy_shlibs_to_remote()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.runCmd("thread step-over", "Stepping over function that loads library")

        # The stop reason should be step end.
        self.expect(
            "thread list",
            "step over succeeded.",
            substrs=["stopped", "stop reason = step over"],
        )

    # We can't find a breakpoint location for d_init before launching because
    # executable dependencies are resolved relative to the debuggers PWD. Bug?
    @expectedFailureAll(
        oslist=["freebsd", "linux", "netbsd"], triple=no_match("aarch64-.*-android")
    )
    @expectedFailureAll(oslist=["windows"], archs=["aarch64"])
    def test_static_init_during_load(self):
        """Test that we can set breakpoints correctly in static initializers"""
        self.copy_shlibs_to_remote()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        a_init_bp_num = lldbutil.run_break_set_by_symbol(
            self, "a_init", num_expected_locations=0
        )
        b_init_bp_num = lldbutil.run_break_set_by_symbol(
            self, "b_init", num_expected_locations=0
        )
        d_init_bp_num = lldbutil.run_break_set_by_symbol(
            self, "d_init", num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=[
                "stopped",
                "d_init",
                "stop reason = breakpoint %d" % d_init_bp_num,
            ],
        )

        self.runCmd("continue")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=[
                "stopped",
                "b_init",
                "stop reason = breakpoint %d" % b_init_bp_num,
            ],
        )
        self.expect("thread backtrace", substrs=["b_init", "dylib_open", "main"])

        self.runCmd("continue")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=[
                "stopped",
                "a_init",
                "stop reason = breakpoint %d" % a_init_bp_num,
            ],
        )
        self.expect("thread backtrace", substrs=["a_init", "dylib_open", "main"])
