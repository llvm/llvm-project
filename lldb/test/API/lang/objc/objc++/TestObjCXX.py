"""
Make sure that ivars of Objective-C++ classes are visible in LLDB.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCXXTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_break(self):
        """Test ivars of Objective-C++ classes"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires Objective-C 2.0 runtime")

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(
            self, 'breakpoint 1', num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expr f->f", "Found ivar in class",
                    substrs=["= 3"])
