"""Test that corefiles with LC_NOTE "addrable bits" load command, creating and reading."""



import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestAddrableBitsCorefile(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def initial_setup(self):
        self.build()
        self.exe = self.getBuildArtifact("a.out")
        self.corefile = self.getBuildArtifact("corefile")

    @skipIf(archs=no_match(['arm64e']))
    @skipUnlessDarwin
    def test_lc_note_addrable_bits(self):
        self.initial_setup()

        self.target = self.dbg.CreateTarget(self.exe)
        err = lldb.SBError()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.c'))
        self.assertEqual(process.IsValid(), True)

        found_main = False
        for f in thread.frames:
          if f.GetFunctionName() == "main":
            found_main = True
        self.assertTrue(found_main)

        cmdinterp = self.dbg.GetCommandInterpreter()
        res = lldb.SBCommandReturnObject()
        cmdinterp.HandleCommand("process save-core %s" % self.corefile, res)
        self.assertTrue(res.Succeeded(), True)
        process.Kill()
        self.dbg.DeleteTarget(target)

        target = self.dbg.CreateTarget('')
        process = target.LoadCore(self.corefile)
        self.assertTrue(process.IsValid(), True)
        thread = process.GetSelectedThread()

        found_main = False
        for f in thread.frames:
          if f.GetFunctionName() == "main":
            found_main = True
        self.assertTrue(found_main)

