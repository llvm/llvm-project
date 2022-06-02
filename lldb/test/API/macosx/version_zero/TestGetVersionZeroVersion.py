"""
Read in a library with a version number of 0.0.0, make sure we produce a good version.
"""



import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestGetVersionForZero(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_get_version_zero(self):
        """Read in a library with a version of 0.0.0.  Test SBModule::GetVersion"""
        self.yaml2obj("libDylib.dylib.yaml", self.getBuildArtifact("libDylib.dylib"))
        self.do_test()

    def do_test(self):
        lib_name = "libDylib.dylib"
        target = lldbutil.run_to_breakpoint_make_target(self, exe_name=lib_name)
        module = target.FindModule(lldb.SBFileSpec(lib_name))
        self.assertTrue(module.IsValid(), "Didn't find the libDylib.dylib module")
        # For now the actual version numbers are wrong for a library of 0.0.0
        # but the previous code would crash iterating over the resultant
        # list.  So we are testing that that doesn't happen.
        did_iterate = False
        for elem in module.GetVersion():
            did_iterate = True
        self.assertTrue(did_iterate, "Didn't get into the GetVersion loop")

