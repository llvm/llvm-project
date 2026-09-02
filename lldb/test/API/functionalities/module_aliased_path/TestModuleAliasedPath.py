"""
Test that a module registered with the target through an aliased directory is
not duplicated when the process loads the same file under its canonical path.

A process reports the canonical path of each image it loads: on Windows the
LOAD_DLL_DEBUG_EVENT path comes from GetFinalPathNameByHandleW, which resolves
symbolic links, junctions and subst/mapped drives. The path LLDB already has for
the same file is whatever it was given, and test harnesses hand it the build
directory as spelled on the command line -- swift-ci builds under a subst drive,
so the module is added as `T:\\...\\foo.dll` while the load event reports
`C:\\Users\\...\\foo.dll`. Matching those textually fails, so LLDB adds a second
module for a file it already has and the first one is left behind, unloaded,
with its breakpoint locations unresolved.
"""

import os
import subprocess
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfTargetDoesNotSupportSharedLibraries()
class TestModuleAliasedPath(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def make_directory_alias(self, link, target):
        """Make `link` a second name for the directory `target`, or return False."""
        if sys.platform == "win32":
            command = 'mklink /J "%s" "%s"' % (link, target)
            if subprocess.call(command, shell=True, stdout=subprocess.DEVNULL) != 0:
                return False
            self.addTearDownHook(lambda: os.rmdir(link))
        else:
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                return False
            self.addTearDownHook(lambda: os.unlink(link))
        return os.path.realpath(link) != link

    @skipIfRemote
    def test_module_added_through_aliased_path(self):
        self.build()

        lib_name = self.platformContext.getFullLibName("foo")
        build_dir = self.getBuildDir()
        alias_dir = build_dir + ".alias"
        if not self.make_directory_alias(alias_dir, build_dir):
            self.skipTest("could not alias the build directory")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.assertTrue(
            target.AddModule(os.path.join(alias_dir, lib_name), None, None).IsValid(),
            "added the library through the aliased path",
        )

        bkpt = target.BreakpointCreateBySourceRegex(
            "break here", lldb.SBFileSpec("foo.c")
        )
        self.assertEqual(bkpt.GetNumLocations(), 1, "one location before launch")

        env = []
        shlib_var = self.platformContext.shlib_environment_var
        if shlib_var and shlib_var != "PATH":
            env.append("%s=%s" % (shlib_var, build_dir))

        process = target.LaunchSimple(
            None, env or None, self.get_process_working_directory()
        )
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, bkpt)
        self.assertIsNotNone(thread, "stopped at the breakpoint in the library")

        modules = [
            m for m in target.module_iter() if m.GetFileSpec().GetFilename() == lib_name
        ]
        self.assertEqual(len(modules), 1, "one module for the library")
        self.assertEqual(bkpt.GetNumLocations(), 1, "one location after launch")
        self.assertEqual(bkpt.GetNumResolvedLocations(), 1)
