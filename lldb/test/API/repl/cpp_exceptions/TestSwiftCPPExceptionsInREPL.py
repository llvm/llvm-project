"""
Make sure we don't trap exceptions with SetREPLEnabled(true) or "lldb --repl"
"""

from __future__ import print_function


import os
import time
import re
import subprocess
import lldb
import swift
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators


class TestSwiftREPLExceptions(TestBase):

    # If your test case doesn't stress debug info, the 
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    def test_set_repl_mode_exceptions(self):
        """ Test that SetREPLMode turns off trapping exceptions."""
        return
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.swift")
        self.do_repl_mode_test()

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    def test_repl_exceptions(self):
        """ Test the lldb --repl turns off trapping exceptions."""
        self.build()
        self.do_repl_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @decorators.skipIfRemote
    def do_repl_test(self):
        sdk_root = ""
        with open(self.getBuildArtifact("sdkroot.txt"), 'r') as f:
            sdk_root = f.readlines()[0]
        self.assertGreater(len(sdk_root), 0)
        build_dir = self.getBuildDir()
        repl_args = [lldbtest_config.lldbExec, "-x", "--repl=-enable-objc-interop -sdk %s -L%s -I%s"%(sdk_root, build_dir, build_dir)]
        repl_proc = subprocess.Popen(repl_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=build_dir)
        input_str = "import Wrapper\ncall_cpp()\n:quit"
        (stdoutdata, stderrdata) = repl_proc.communicate(input=input_str.encode())
        stdoutdata = stdoutdata.decode("utf-8") if stdoutdata else None
        stderrdata = stderrdata.decode("utf-8") if stderrdata else None
        self.assertTrue("I called it successfully" in stdoutdata, "Didn't call call_cpp successfully: out: \n%s\nerr: %s"%(stdoutdata, stderrdata))
        
    def do_repl_mode_test(self):
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        self.registerSharedLibrariesWithTarget(target, ['Wrapper'])

        if lldb.remote_platform:
            wd = lldb.remote_platform.GetWorkingDirectory()
            filename = 'libCppLib.dylib'
            err = lldb.remote_platform.Put(
                lldb.SBFileSpec(self.getBuildArtifact(filename)),
                lldb.SBFileSpec(os.path.join(wd, filename)))
            self.assertFalse(err.Fail(), 'Failed to copy ' + filename)
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file) 

        frame = thread.GetFrameAtIndex(0)
        options = lldb.SBExpressionOptions()
        options.SetREPLMode(True)
        val = frame.EvaluateExpression("call_cpp(); 5", options)
        self.assertSuccess(val.GetError(), "Got an error evaluating expression")
        self.assertEqual(val.GetValueAsUnsigned(), 5,"The expression didn't return the correct result")
