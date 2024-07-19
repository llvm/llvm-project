"""Test binaries with delay-init dependencies."""

import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDelayInitDependencies(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "15.0"])
    def test_delay_init_dependency(self):
        TestBase.setUp(self)
        out = subprocess.run(
            ["xcrun", "ld", "-delay_library"],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if "delay_library missing" not in out.stderr:
            self.skipTest(
                "Skipped because the linker doesn't know about -delay_library"
            )
        self.build()
        main_source = "main.c"
        exe = self.getBuildArtifact("a.out")
        lib = self.getBuildArtifact("libfoo.dylib")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # libfoo.dylib should not be in the target pre-execution
        for m in target.modules:
            self.assertNotEqual(m.GetFileSpec().GetFilename(), "libfoo.dylib")

        # This run without arguments will not load libfoo.dylib
        li = lldb.SBLaunchInfo([])
        li.SetWorkingDirectory(self.getBuildDir())
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c"), li
        )
        for m in target.modules:
            self.assertNotEqual(m.GetFileSpec().GetFilename(), "libfoo.dylib")

        process.Kill()
        self.dbg.DeleteTarget(target)

        # This run with one argument will load libfoo.dylib
        li = lldb.SBLaunchInfo([])
        li.SetWorkingDirectory(self.getBuildDir())
        li.SetArguments(["one-argument"], True)
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c"), li
        )

        found_libfoo = False
        for m in target.modules:
            if m.GetFileSpec().GetFilename() == "libfoo.dylib":
                found_libfoo = True
        self.assertTrue(found_libfoo)
