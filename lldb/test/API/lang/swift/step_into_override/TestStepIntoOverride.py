import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import platform
import unittest2

class TestStepIntoOverride(lldbtest.TestBase):

    @swiftTest
    def test_swift_stepping(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        # If you are running against a debug swift you are going to
        # end up stepping into the stdlib and that will make stepping
        # tests impossible to write.  So avoid that.

        if platform.system() == 'Darwin':
            lib_name = "libswiftCore.dylib"
        else:
            lib_name = "libswiftCore.so"

        self.dbg.HandleCommand(
            "settings set "
            "target.process.thread.step-avoid-libraries {}".format(lib_name))

    def do_test(self):
        """Tests that we can step reliably in swift code."""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        target, process, thread, breakpoint = lldbutil.run_to_source_breakpoint(self, 'break here', self.main_source_spec)

        # Step into the function.
        thread.StepInto()
        frame = thread.frames[0]

        # Make sure we step into the right function, the one that takes
        # an Optional<Int> as argument.
        func = frame.GetFunctionName()
        self.assertEqual(func, 'a.Base.foo(Swift.Optional<Swift.Int>) -> ()')

        # And that we can find the value of `a` after we stepped in.
        valobj = frame.FindVariable('a')
        self.assertEqual(valobj.GetSummary(), '3')
