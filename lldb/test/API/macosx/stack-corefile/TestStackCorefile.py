"""Test that lldb can create a stack-only corefile, and load the main binary."""

import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestStackCorefile(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    @skipUnlessDarwin
    def test(self):

        corefile = self.getBuildArtifact("process.core")
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c"))

        frame = thread.GetFrameAtIndex(0)
        stack_int = frame.GetValueForVariablePath("stack_int")
        heap_int = frame.GetValueForVariablePath("*heap_int")
        stack_str = frame.GetValueForVariablePath("stack_str")
        heap_str = frame.GetValueForVariablePath("heap_str")
        self.assertEqual(stack_int.GetValueAsUnsigned(), 5)
        self.assertEqual(heap_int.GetValueAsUnsigned(), 10)
        self.assertEqual(stack_str.summary, '"stack string"')
        self.assertEqual(heap_str.summary, '"heap string"')

        self.runCmd("process save-core -s stack " + corefile)
        process.Kill()
        self.dbg.DeleteTarget(target)

        # Now load the corefile
        target = self.dbg.CreateTarget('')
        process = target.LoadCore(corefile)
        thread = process.GetSelectedThread()
        self.assertTrue(process.IsValid())
        self.assertTrue(process.GetSelectedThread().IsValid())
        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("bt")
            self.runCmd("fr v")
        num_modules = target.GetNumModules()
        #  We should only have the a.out binary and possibly
        # the libdyld.dylib.  Extra libraries loaded means 
        # extra LC_NOTE and unnecessarily large corefile.
        self.assertTrue(num_modules == 1 or num_modules == 2)

        # The heap variables should be unavailable now.
        frame = thread.GetFrameAtIndex(0)
        stack_int = frame.GetValueForVariablePath("stack_int")
        heap_int = frame.GetValueForVariablePath("*heap_int")
        stack_str = frame.GetValueForVariablePath("stack_str")
        heap_str = frame.GetValueForVariablePath("heap_str")

        ## The heap SBValues both come back as IsValid()==true,
        ## which I'm not so sure is a great/correct thing --
        ## it hides the memory read error that actually happened,
        ## and we don't have a real value.
        self.assertEqual(stack_int.GetValueAsUnsigned(), 5)
        self.assertEqual(heap_int.GetValueAsUnsigned(), 0)
        self.assertEqual(stack_str.summary, '"stack string"')
        self.assertEqual(heap_str.summary, '""')
