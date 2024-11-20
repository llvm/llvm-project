"""
Test that SBProcess.LoadImageUsingPaths uses RTLD_LAZY
"""


import os
import shutil
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LoadUsingLazyBind(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipIfWindows  # The Windows platform doesn't implement DoLoadImage.
    @skipIf(oslist=["linux"], archs=["arm"])  # Fails on arm/linux
    # Failing for unknown reasons on Linux, see
    # https://bugs.llvm.org/show_bug.cgi?id=49656.
    def test_load_using_lazy_bind(self):
        """Test that we load using RTLD_LAZY"""

        self.build()
        wd = os.path.realpath(self.getBuildDir())

        def make_lib_name(name):
            return (
                self.platformContext.shlib_prefix
                + name
                + "."
                + self.platformContext.shlib_extension
            )

        def make_lib_path(name):
            libpath = os.path.join(wd, make_lib_name(name))
            self.assertTrue(os.path.exists(libpath))
            return libpath

        libt2_0 = make_lib_path("t2_0")
        libt2_1 = make_lib_path("t2_1")

        # Overwrite t2_0 with t2_1 to delete the definition of `use`.
        shutil.copy(libt2_1, libt2_0)

        # Launch a process and break
        (target, process, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp"), extra_images=["t1"]
        )

        # Load libt1; should fail unless we use RTLD_LAZY
        error = lldb.SBError()
        lib_spec = lldb.SBFileSpec(make_lib_name("t1"))
        paths = lldb.SBStringList()
        paths.AppendString(wd)
        out_spec = lldb.SBFileSpec()
        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)
        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token")

        # Calling `f1()` should return 5.
        frame = thread.GetFrameAtIndex(0)
        val = frame.EvaluateExpression("f1()")
        self.assertTrue(val.IsValid())
        self.assertEqual(val.GetValueAsSigned(-1), 5)
