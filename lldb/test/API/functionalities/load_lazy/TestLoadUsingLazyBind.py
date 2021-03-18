"""
Test that SBProcess.LoadImageUsingPaths uses RTLD_LAZY
"""



import os
import shutil
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfRemote
@skipIfWindows # The Windows platform doesn't implement DoLoadImage.
class LoadUsingLazyBind(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Invoke the default build rule.
        self.build()

        self.wd = os.path.realpath(self.getBuildDir())

        self.ext = 'so'
        if self.platformIsDarwin():
            self.ext = 'dylib'

        # Overwrite t2_0 with t2_1 to delete the definition of `use`.
        shutil.copy(os.path.join(self.wd, 'libt2_1.{}'.format(self.ext)),
                    os.path.join(self.wd, 'libt2_0.{}'.format(self.ext)))

    @skipIfRemote
    @skipIfWindows # The Windows platform doesn't implement DoLoadImage.
    def test_load_using_lazy_bind(self):
        """Test that we load using RTLD_LAZY"""

        (target, process, thread, _) = lldbutil.run_to_source_breakpoint(self,
                                                "break here",
                                                lldb.SBFileSpec("main.cpp"))
        error = lldb.SBError()
        lib_spec = lldb.SBFileSpec("libt1.{}".format(self.ext))
        paths = lldb.SBStringList()
        paths.AppendString(self.wd)
        out_spec = lldb.SBFileSpec()
        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)
        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token")
