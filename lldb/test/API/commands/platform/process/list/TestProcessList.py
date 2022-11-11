"""
Test process list.
"""


import os
import lldb
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessListTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows  # https://bugs.llvm.org/show_bug.cgi?id=43702
    @skipIfRemote   # rdar://problem/66542336
    def test_process_list_with_args(self):
        """Test process list show process args"""
        self.build()
        exe = self.getBuildArtifact("TestProcess")

        # Spawn a new process
        sync_file = lldbutil.append_to_process_working_directory(self,
                "ready.txt")
        popen = self.spawnSubprocess(exe, args=[sync_file, "arg1", "--arg2", "arg3"])
        lldbutil.wait_for_file_on_target(self, sync_file)

        substrs = [str(popen.pid), "TestProcess", "arg1 --arg2 arg3"]
        self.expect("platform process list -v", substrs=substrs)
