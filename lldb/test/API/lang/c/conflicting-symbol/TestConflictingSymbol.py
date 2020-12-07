"""Test that conflicting symbols in different shared libraries work correctly"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestConflictingSymbols(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        lldbutil.mkdir_p(self.getBuildArtifact("One"))
        lldbutil.mkdir_p(self.getBuildArtifact("Two"))

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48416")
    @expectedFailureNetBSD
    def test_conflicting_symbols(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, ['One', 'Two'])

        lldbutil.run_break_set_by_source_regexp(self, '// break here',
                extra_options='-f One.c', num_expected_locations=-2)
        lldbutil.run_break_set_by_source_regexp(self, '// break here',
                extra_options='-f Two.c', num_expected_locations=-2)
        lldbutil.run_break_set_by_source_regexp(self, '// break here',
                extra_options='-f main.c', num_expected_locations=1)

        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        self.expect(
            "expr (unsigned long long)conflicting_symbol",
            "Symbol from One should be found",
            substrs=[
                "11111"])

        self.runCmd("continue", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect(
            "expr (unsigned long long)conflicting_symbol",
            "Symbol from Two should be found",
            substrs=[
                "22222"])

        self.runCmd("continue", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect(
            "expr (unsigned long long)conflicting_symbol",
            "An error should be printed when symbols can't be ordered",
            error=True,
            substrs=[
                "Multiple internal symbols"])

    @expectedFailureAll(bugnumber="llvm.org/pr35043")
    @skipIfWindows # This test is "passing" on Windows, but it is a false positive.
    def test_shadowed(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, ['One', 'Two'])

        lldbutil.run_break_set_by_source_regexp(self, '// break here',
                extra_options='-f main.c', num_expected_locations=1)

        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # As we are shadowing the conflicting symbol, there should be no
        # ambiguity in this expression.
        self.expect(
            "expr int conflicting_symbol = 474747; conflicting_symbol",
            substrs=[ "474747"])
