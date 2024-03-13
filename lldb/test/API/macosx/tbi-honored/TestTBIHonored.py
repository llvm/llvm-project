"""Test that lldb on Darwin ignores metadata in the top byte of addresses."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTBIHonored(TestBase):
    @no_debug_info_test
    @skipUnlessDarwin
    @skipIf(archs=no_match(["arm64", "arm64e"]))
    @skipIfRemote
    def do_variable_access_tests(self, frame):
        self.assertEqual(
            frame.variables["pb"][0]
            .GetChildMemberWithName("p")
            .Dereference()
            .GetValueAsUnsigned(),
            15,
        )
        addr = frame.variables["pb"][0].GetChildMemberWithName("p").GetValueAsUnsigned()
        self.expect("expr -- *pb.p", substrs=["15"])
        self.expect("frame variable *pb.p", substrs=["15"])
        self.expect("expr -- *(int*)0x%x" % addr, substrs=["15"])

    def test(self):
        corefile = self.getBuildArtifact("process.core")
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.do_variable_access_tests(thread.GetFrameAtIndex(0))

        self.runCmd("process save-core -s stack " + corefile)
        self.dbg.DeleteTarget(target)

        # Now load the corefile
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(corefile)
        thread = process.GetSelectedThread()
        self.assertTrue(process.GetSelectedThread().IsValid())

        self.do_variable_access_tests(thread.GetFrameAtIndex(0))
