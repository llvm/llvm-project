"""
Test that a process launched through an aliased directory reports its modules
under that directory, so LLDB does not duplicate the modules it already has.

The target is created from a junction to the build directory. lldb-server must
report the executable and the DLL next to it with the path the Windows loader
recorded, which keeps the junction, rather than with the resolved path, which
does not. Otherwise LLDB adds a second module for each file and the breakpoints
get a location in both.
"""

import os
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireWindows
@skipIfWindowsAndNoLLDBServer
@skipIfRemote
class TestModuleAliasedPath(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def make_junction(self, real_dir):
        """Create a junction to real_dir, skipping the test if the host can't
        create one."""
        alias_dir = real_dir + ".alias"
        try:
            if os.path.lexists(alias_dir):
                os.rmdir(alias_dir)
            subprocess.run(
                'mklink /J "%s" "%s"' % (alias_dir, real_dir),
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            detail = getattr(e, "stderr", None) or e
            self.skipTest("could not create a junction: %s" % detail)
        self.addTearDownHook(lambda: os.rmdir(alias_dir))
        self.assertNotEqual(os.path.realpath(alias_dir), alias_dir)
        return alias_dir

    def test_launch_through_aliased_path(self):
        self.build()

        alias_dir = self.make_junction(self.getBuildDir())

        target = self.dbg.CreateTarget(os.path.join(alias_dir, "a.out"))
        self.assertTrue(target, VALID_TARGET)
        main_bkpt = target.BreakpointCreateBySourceRegex(
            "break main", lldb.SBFileSpec("main.c")
        )
        foo_bkpt = target.BreakpointCreateBySourceRegex(
            "break here", lldb.SBFileSpec("foo.c")
        )
        self.assertEqual(main_bkpt.GetNumLocations(), 1)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint(process, main_bkpt)
        )

        lib_name = self.platformContext.getFullLibName("foo")
        for name in ["a.out", lib_name]:
            paths = [
                m.GetFileSpec().fullpath
                for m in target.module_iter()
                if m.GetFileSpec().GetFilename() == name
            ]
            self.assertEqual(len(paths), 1, "one module for %s: %s" % (name, paths))
            self.assertTrue(
                paths[0].lower().startswith(alias_dir.lower()),
                "%s is reported under the junction" % paths[0],
            )

        for bkpt in [main_bkpt, foo_bkpt]:
            self.assertEqual(bkpt.GetNumLocations(), 1)
            self.assertEqual(bkpt.GetNumResolvedLocations(), 1)

        process.Continue()
        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint(process, foo_bkpt)
        )
