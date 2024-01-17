# This test checks that we make the executable the first
# element in the image list.

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExecutableIsFirst(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # ELF does not have a hard distinction between shared libraries and
    # (position-independent) executables
    @skipIf(oslist=no_match(lldbplatformutil.getDarwinOSTriples()+["windows"]))
    def test_executable_is_first_before_run(self):
        self.build()

        ctx = self.platformContext
        lib_name = ctx.shlib_prefix + "bar." + ctx.shlib_extension

        exe = self.getBuildArtifact("a.out")
        lib = self.getBuildArtifact(lib_name)

        target = self.dbg.CreateTarget(None)
        module = target.AddModule(lib, None, None)
        self.assertTrue(module.IsValid(), "Added the module for the library")

        module = target.AddModule(exe, None, None)
        self.assertTrue(module.IsValid(), "Added the executable module")

        # This is the executable module so it should be the first in the list:
        first_module = target.GetModuleAtIndex(0)
        print("This is the first test, this one succeeds")
        self.assertEqual(module, first_module, "This executable is the first module")

        # The executable property is an SBFileSpec to the executable.  Make sure
        # that is also right:
        executable_module = target.executable
        self.assertEqual(
            first_module.file, executable_module, "Python property is also our module"
        )

    def test_executable_is_first_during_run(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break after function call", lldb.SBFileSpec("main.cpp"),
            extra_images=["bar"]
        )

        first_module = target.GetModuleAtIndex(0)
        self.assertTrue(first_module.IsValid(), "We have at least one module")
        executable_module = target.executable
        self.assertEqual(first_module.file, executable_module, "They are the same")
