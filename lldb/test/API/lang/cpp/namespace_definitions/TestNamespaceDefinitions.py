"""Test that forward declarations don't cause bogus conflicts in namespaced types"""



import unittest2
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class NamespaceDefinitionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        bugnumber="llvm.org/pr28948",
        compiler="gcc",
        compiler_version=[
            "<",
            "4.9"])
    @expectedFailureAll(
        bugnumber="llvm.org/pr28948",
        oslist=['linux'], compiler="gcc", archs=['arm','aarch64'])
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureNetBSD
    def test_expr(self):
        self.build()
        self.common_setup()

        self.expect(
            "expression -- Foo::MyClass()",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=['thing = '])

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.cpp'
        self.line = line_number(self.source, '// Set breakpoint here')
        self.shlib_names = ["a", "b"]

    def common_setup(self):
        # Run in synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, self.shlib_names)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])
