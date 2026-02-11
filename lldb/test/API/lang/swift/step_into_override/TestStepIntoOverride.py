import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import platform

class TestStepIntoOverride(lldbtest.TestBase):
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test_swift_stepping(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        lldbutil.ignore_swift_stdlib_when_stepping(platform, self)

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
