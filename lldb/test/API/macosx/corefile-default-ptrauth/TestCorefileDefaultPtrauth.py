"""Test that lldb has a default mask for addressable bits on Darwin arm64 ABI"""


import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCorefileDefaultPtrauth(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is looking explicitly for a dSYM")
    @skipIf(archs=no_match(['arm64','arm64e']))
    @skipUnlessDarwin
    def test_lc_note(self):
        self.build()
        self.test_exe = self.getBuildArtifact("a.out")
        self.create_corefile = self.getBuildArtifact("create-corefile")
        self.corefile = self.getBuildArtifact("core")

        ### Create our corefile
        retcode = call(self.create_corefile + " " +  self.test_exe + " " + self.corefile, shell=True)

        ## This corefile has no metadata telling us how many bits are
        ## used for ptrauth signed function pointers.  We will need lldb
        ## to fall back on its old default value for Darwin arm64 ABIs
        ## to correctly strip the bits.

        self.target = self.dbg.CreateTarget('')
        err = lldb.SBError()
        self.process = self.target.LoadCore(self.corefile)
        self.assertEqual(self.process.IsValid(), True)
        modspec = lldb.SBModuleSpec()
        modspec.SetFileSpec(lldb.SBFileSpec(self.test_exe, True))
        m = self.target.AddModule(modspec)
        self.assertTrue(m.IsValid())
        self.target.SetModuleLoadAddress (m, 0)
        
        # target variable should show us both the actual function
        # pointer with ptrauth bits and the symbol it resolves to,
        # with the ptrauth bits stripped, e.g.
        #  (int (*)(...)) fmain = 0xe46bff0100003f90 (actual=0x0000000100003f90 a.out`main at main.c:3)

        self.expect("target variable fmain", substrs=['fmain = 0x', '(actual=0x', 'main at main.c'])
