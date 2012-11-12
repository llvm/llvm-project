"""
Test that breakpoint by symbol name works correctly with dynamic libs.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LoadUnloadTestCase(TestBase):

    mydir = os.path.join("functionalities", "load_unload")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.c',
                                '// Set break point at this line for test_lldb_process_load_and_unload_commands().')
        self.line_d_function = line_number('d.c',
                                           '// Find this line number within d_dunction().')

    def test_modules_search_paths(self):
        """Test target modules list after loading a different copy of the library libd.dylib, and verifies that it works with 'target modules search-paths add'."""

        # Invoke the default build rule.
        self.buildDefault()

        if sys.platform.startswith("darwin"):
            dylibName = 'libd.dylib'
            dylibPath = 'DYLD_LIBRARY_PATH'

        # The directory with the dynamic library we did not link to.
        new_dir = os.path.join(os.getcwd(), "hidden")

        old_dylib = os.path.join(os.getcwd(), dylibName)
        new_dylib = os.path.join(new_dir, dylibName)

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("target modules list",
            substrs = [old_dylib])
        #self.expect("target modules list -t 3",
        #    patterns = ["%s-[^-]*-[^-]*" % self.getArchitecture()])
        # Add an image search path substitution pair.
        self.runCmd("target modules search-paths add %s %s" % (os.getcwd(), new_dir))
        # Add teardown hook to clear image-search-paths after the test.
        # rdar://problem/10501020
        # Uncomment the following to reproduce 10501020.
        self.addTearDownHook(lambda: self.runCmd("target modules search-paths clear"))

        self.expect("target modules search-paths list",
            substrs = [os.getcwd(), new_dir])

        self.expect("target modules search-paths query %s" % os.getcwd(), "Image search path successfully transformed",
            substrs = [new_dir])

        # Obliterate traces of libd from the old location.
        os.remove(old_dylib)
        # Inform dyld of the new path, too.
        env_cmd_string = "settings set target.env-vars " + dylibPath + "=" + new_dir
        if self.TraceOn():
            print "Set environment to: ", env_cmd_string
        self.runCmd(env_cmd_string)
        self.runCmd("settings show target.env-vars")

        remove_dyld_path_cmd = "settings remove target.env-vars " + dylibPath
        self.addTearDownHook(lambda: self.runCmd(remove_dyld_path_cmd))

        self.runCmd("run")

        self.expect("target modules list", "LLDB successfully locates the relocated dynamic library",
            substrs = [new_dylib])

        
    def test_dyld_library_path(self):
        """Test DYLD_LIBRARY_PATH after moving libd.dylib, which defines d_function, somewhere else."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        if sys.platform.startswith("darwin"):
            dylibName = 'libd.dylib'
            dsymName = 'libd.dylib.dSYM'
            dylibPath = 'DYLD_LIBRARY_PATH'

        # The directory to relocate the dynamic library and its debugging info.
        special_dir = "hidden"
        new_dir = os.path.join(os.getcwd(), special_dir)

        old_dylib = os.path.join(os.getcwd(), dylibName)
        new_dylib = os.path.join(new_dir, dylibName)
        old_dSYM = os.path.join(os.getcwd(), dsymName)
        new_dSYM = os.path.join(new_dir, dsymName)

        #system(["ls", "-lR", "."])

        # Try running with the DYLD_LIBRARY_PATH environment variable set, make sure
        # we pick up the hidden dylib.

        env_cmd_string = "settings set target.env-vars " + dylibPath + "=" + new_dir
        if self.TraceOn():
            print "Set environment to: ", env_cmd_string
        self.runCmd(env_cmd_string)
        self.runCmd("settings show target.env-vars")

        remove_dyld_path_cmd = "settings remove target.env-vars " + dylibPath
        self.addTearDownHook(lambda: self.runCmd(remove_dyld_path_cmd))

        lldbutil.run_break_set_by_file_and_line (self, "d.c", self.line_d_function, num_expected_locations=1, loc_exact=True)

        # For now we don't track DYLD_LIBRARY_PATH, so the old library will be in
        # the modules list.
        self.expect("target modules list",
            substrs = [os.path.basename(old_dylib)],
            matching=True)

        self.runCmd("run")
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            patterns = ["frame #0.*d_function.*at d.c:%d" % self.line_d_function])

        # After run, make sure the hidden library is present, and the one we didn't 
        # load is not.
        self.expect("target modules list",
            substrs = [special_dir, os.path.basename(new_dylib)])

    def test_lldb_process_load_and_unload_commands(self):
        """Test that lldb process load/unload command work correctly."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break at main.c before the call to dlopen().
        # Use lldb's process load command to load the dylib, instead.

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Make sure that a_function does not exist at this point.
        self.expect("image lookup -n a_function", "a_function should not exist yet",
                    error=True, matching=False,
            patterns = ["1 match found .* %s" % self.mydir])

        # Use lldb 'process load' to load the dylib.
        self.expect("process load liba.dylib", "liba.dylib loaded correctly",
            patterns = ['Loading "liba.dylib".*ok',
                        'Image [0-9]+ loaded'])

        # Search for and match the "Image ([0-9]+) loaded" pattern.
        output = self.res.GetOutput()
        pattern = re.compile("Image ([0-9]+) loaded")
        for l in output.split(os.linesep):
            #print "l:", l
            match = pattern.search(l)
            if match:
                break
        index = match.group(1)

        # Now we should have an entry for a_function.
        self.expect("image lookup -n a_function", "a_function should now exist",
            patterns = ["1 match found .*%s" % self.mydir])

        # Use lldb 'process unload' to unload the dylib.
        self.expect("process unload %s" % index, "liba.dylib unloaded correctly",
            patterns = ["Unloading .* with index %s.*ok" % index])

        self.runCmd("process continue")

    def test_load_unload(self):
        """Test breakpoint by name works correctly with dlopen'ing."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        lldbutil.run_break_set_by_symbol (self, "a_function", num_expected_locations=0)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'a_function',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        # Issue the 'contnue' command.  We should stop agaian at a_function.
        # The stop reason of the thread should be breakpoint and at a_function.
        self.runCmd("continue")

        # rdar://problem/8508987
        # The a_function breakpoint should be encountered twice.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'a_function',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 2.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 2'])

    def test_step_over_load (self):
        """Test stepping over code that loads a shared library works correctly."""

        # Invoke the default build rule.
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break by function name a_function (not yet loaded).
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint and at a_function.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.runCmd("thread step-over", "Stepping over function that loads library")
        
        # The stop reason should be step end.
        self.expect("thread list", "step over succeeded.", 
            substrs = ['stopped',
                      'stop reason = step over'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
