import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestArchetypeInConditionalBreakpoint(TestBase):
    @swiftTest
    def test_stops_free_function(self):
        self.stops("break here for free function")

    @swiftTest
    def test_doesnt_stop_free_function(self):
        self.doesnt_stop("break here for free function")

    @swiftTest
    def test_stops_class(self):
        self.stops("break here for class")

    @swiftTest
    def test_doesnt_stop_class(self):
        self.doesnt_stop("break here for class")

    def stops(self, breakpoint_string):
        """Tests that using archetypes in a conditional breakpoint's expression works correctly"""
        self.build()
        target = lldbutil.run_to_breakpoint_make_target(self)

        breakpoint = target.BreakpointCreateBySourceRegex(
            breakpoint_string, lldb.SBFileSpec("main.swift")
        )

        breakpoint.SetCondition("T.self == Int.self")
        _, process, _, _ = lldbutil.run_to_breakpoint_do_run(self, target, breakpoint)

        self.assertEqual(process.state, lldb.eStateStopped)
        self.expect("expression T.self", substrs=["Int"])

    def doesnt_stop(self, breakpoint_string):
        """Tests that using archetypes in a conditional breakpoint's expression works correctly"""
        self.build()
        target = lldbutil.run_to_breakpoint_make_target(self)

        breakpoint = target.BreakpointCreateBySourceRegex(
            breakpoint_string, lldb.SBFileSpec("main.swift")
        )

        breakpoint.SetCondition("T.self == Double.self")

        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())

        error = lldb.SBError()
        process = target.Launch(launch_info, error)

        # Make sure that we didn't stop since the condition doesn't match
        self.assertEqual(process.state, lldb.eStateExited)
