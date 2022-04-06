from __future__ import print_function
import lldb
import os
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


@skipIfRemote
class TestQemuAPI(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_file_api(self):
        qemu = lldb.SBPlatform("qemu-user")
        host = lldb.SBPlatform.GetHostPlatform()

        target = self.getBuildArtifact("target.c")
        main_c = lldb.SBFileSpec(self.getSourcePath("main.c"))

        self.assertSuccess(qemu.Put(main_c, lldb.SBFileSpec(target)))
        self.assertTrue(os.path.exists(target))
        self.assertEqual(qemu.GetFilePermissions(target),
                host.GetFilePermissions(target))

        self.assertSuccess(qemu.MakeDirectory(
            self.getBuildArtifact("target_dir")))
        self.assertTrue(os.path.isdir(self.getBuildArtifact("target_dir")))
