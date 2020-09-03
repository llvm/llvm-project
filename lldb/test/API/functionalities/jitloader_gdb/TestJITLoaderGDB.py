"""Test for the JITLoaderGDB interface"""


import unittest2
import os
import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class JITLoaderGDBTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipTestIfFn(
        lambda: "Skipped because the test crashes the test runner",
        bugnumber="llvm.org/pr24702")
    @unittest2.expectedFailure("llvm.org/pr24702")
    def test_bogus_values(self):
        """Test that we handle inferior misusing the GDB JIT interface"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The inferior will now pass bogus values over the interface. Make sure
        # we don't crash.
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    def gen_log_file(self):
        logfile = self.getBuildArtifact("jitintgdb-{}.txt".format(self.getArchitecture()))
        def cleanup():
            if os.path.exists(logfile):
                os.unlink(logfile)
        self.addTearDownHook(cleanup)
        return logfile

    def test_jit_int_default(self):
        self.expect("settings show plugin.jit-loader.gdb.enable",
                    substrs=["plugin.jit-loader.gdb.enable (enum) = default"])

    @skipIfWindows # This test fails on Windows during C code build
    def test_jit_int_on(self):
        """Tests interface with 'enable' settings 'on'"""
        self.build()
        exe = self.getBuildArtifact("simple")

        logfile = self.gen_log_file()
        self.runCmd("log enable -f %s lldb jit" % (logfile))
        self.runCmd("settings set plugin.jit-loader.gdb.enable on")
        def cleanup():
            self.runCmd("log disable lldb")
            self.runCmd("settings set plugin.jit-loader.gdb.enable default")
        self.addTearDownHook(cleanup)

        # Launch the process.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        if not configuration.is_reproducer():
            self.assertTrue(os.path.exists(logfile))
            logcontent = open(logfile).read()
            self.assertIn("SetJITBreakpoint setting JIT breakpoint", logcontent)

    @skipIfWindows # This test fails on Windows during C code build
    def test_jit_int_off(self):
        """Tests interface with 'enable' settings 'off'"""
        self.build()
        exe = self.getBuildArtifact("simple")

        logfile = self.gen_log_file()
        self.runCmd("log enable -f %s lldb jit" % (logfile))
        self.runCmd("settings set plugin.jit-loader.gdb.enable off")
        def cleanup():
            self.runCmd("log disable lldb")
            self.runCmd("settings set plugin.jit-loader.gdb.enable default")
        self.addTearDownHook(cleanup)

        # Launch the process.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        if not configuration.is_reproducer():
            self.assertTrue(os.path.exists(logfile))
            logcontent = open(logfile).read()
            self.assertNotIn("SetJITBreakpoint setting JIT breakpoint", logcontent)
