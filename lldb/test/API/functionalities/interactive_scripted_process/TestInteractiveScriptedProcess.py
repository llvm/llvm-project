"""
Test the functionality of interactive scripted processes
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import json, os


class TestInteractiveScriptedProcess(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Build and load test program
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.script_module = "interactive_scripted_process"
        self.script_file = self.script_module + ".py"

    # These tests are flakey and sometimes timeout.  They work most of the time
    # so the basic event flow is right, but somehow the handling is off.
    @skipUnlessDarwin
    @skipIfDarwin
    def test_passthrough_launch(self):
        """Test a simple pass-through process launch"""
        self.passthrough_launch()

        lldbutil.run_break_set_by_source_regexp(self, "also break here")
        self.assertEqual(self.mux_target.GetNumBreakpoints(), 2)
        error = self.mux_process.Continue()
        self.assertSuccess(error, "Resuming multiplexer scripted process")
        self.assertTrue(self.mux_process.IsValid(), "Got a valid process")

        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), self.mux_process.GetBroadcaster(), timeout=20
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)
        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

    @skipUnlessDarwin
    @skipIfDarwin
    def test_multiplexed_launch(self):
        """Test a multiple interactive scripted process debugging"""
        self.passthrough_launch()
        self.assertEqual(self.dbg.GetNumTargets(), 2)

        driving_target = self.mux_process.GetScriptedImplementation().driving_target
        self.assertTrue(driving_target.IsValid(), "Driving target is invalid")

        # Create a target for the multiplexed even scripted process
        even_target = self.duplicate_target(driving_target)
        self.assertTrue(
            even_target.IsValid(),
            "Couldn't duplicate driving target to launch multiplexed even scripted process",
        )

        class_name = f"{self.script_module}.MultiplexedScriptedProcess"
        dictionary = {"driving_target_idx": self.dbg.GetIndexOfTarget(self.mux_target)}

        dictionary["parity"] = 0
        muxed_launch_info = self.get_launch_info(class_name, dictionary)

        # Launch Even Child Scripted Process
        error = lldb.SBError()
        even_process = even_target.Launch(muxed_launch_info, error)
        self.assertTrue(
            even_process, "Couldn't launch multiplexed even scripted process"
        )
        self.multiplex(even_process)

        # Check that the even process started running
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), even_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)
        # Check that the even process stopped
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), even_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

        self.assertTrue(even_process.IsValid(), "Got a valid process")
        self.assertState(
            even_process.GetState(), lldb.eStateStopped, "Process is stopped"
        )

        # Create a target for the multiplexed odd scripted process
        odd_target = self.duplicate_target(driving_target)
        self.assertTrue(
            odd_target.IsValid(),
            "Couldn't duplicate driving target to launch multiplexed odd scripted process",
        )

        dictionary["parity"] = 1
        muxed_launch_info = self.get_launch_info(class_name, dictionary)

        # Launch Odd Child Scripted Process
        error = lldb.SBError()
        odd_process = odd_target.Launch(muxed_launch_info, error)
        self.assertTrue(odd_process, "Couldn't launch multiplexed odd scripted process")
        self.multiplex(odd_process)

        # Check that the odd process started running
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), odd_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)
        # Check that the odd process stopped
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), odd_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

        self.assertTrue(odd_process.IsValid(), "Got a valid process")
        self.assertState(
            odd_process.GetState(), lldb.eStateStopped, "Process is stopped"
        )

        # Set a breakpoint on the odd child process
        bkpt = odd_target.BreakpointCreateBySourceRegex(
            "also break here", self.main_source_file
        )
        self.assertEqual(odd_target.GetNumBreakpoints(), 1)
        self.assertTrue(bkpt, "Second breakpoint set on child scripted process")
        self.assertEqual(bkpt.GetNumLocations(), 1, "Second breakpoint has 1 location")

        # Verify that the breakpoint was also set on the multiplexer & real target
        self.assertEqual(self.mux_target.GetNumBreakpoints(), 2)
        bkpt = self.mux_target.GetBreakpointAtIndex(1)
        self.assertEqual(
            bkpt.GetNumLocations(), 1, "Second breakpoint set on mux scripted process"
        )
        self.assertTrue(bkpt.MatchesName("multiplexed_scripted_process_421"))

        self.assertGreater(driving_target.GetNumBreakpoints(), 1)

        # Resume execution on child process
        error = odd_process.Continue()
        self.assertSuccess(error, "Resuming odd child scripted process")
        self.assertTrue(odd_process.IsValid(), "Got a valid process")

        # Since all the execution is asynchronous, the order in which events
        # arrive is non-deterministic, so we need a data structure to make sure
        # we received both the running and stopped event for each target.

        # Initialize the execution event "bingo book", that maps a process index
        # to a dictionary that contains flags that are not set for the process
        # events that we care about (running & stopped)

        execution_events = {
            1: {lldb.eStateRunning: False, lldb.eStateStopped: False},
            2: {lldb.eStateRunning: False, lldb.eStateStopped: False},
            3: {lldb.eStateRunning: False, lldb.eStateStopped: False},
        }

        def fetch_process_event(self, execution_events):
            event = lldbutil.fetch_next_event(
                self,
                self.dbg.GetListener(),
                lldb.SBProcess.GetBroadcasterClass(),
                match_class=True,
            )
            state = lldb.SBProcess.GetStateFromEvent(event)
            self.assertIn(state, [lldb.eStateRunning, lldb.eStateStopped])
            event_process = lldb.SBProcess.GetProcessFromEvent(event)
            self.assertTrue(event_process.IsValid())
            event_target = event_process.GetTarget()
            event_target_idx = self.dbg.GetIndexOfTarget(event_target)
            self.assertFalse(
                execution_events[event_target_idx][state],
                "Event already received for this process",
            )
            execution_events[event_target_idx][state] = True

        for _ in range((self.dbg.GetNumTargets() - 1) * 2):
            fetch_process_event(self, execution_events)

        for target_index, event_states in execution_events.items():
            for state, is_set in event_states.items():
                self.assertTrue(is_set, f"Target {target_index} has state {state} set")

        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)

        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

    def duplicate_target(self, driving_target):
        exe = driving_target.executable.fullpath
        triple = driving_target.triple
        return self.dbg.CreateTargetWithFileAndTargetTriple(exe, triple)

    def get_launch_info(self, class_name, script_dict):
        structured_data = lldb.SBStructuredData()
        structured_data.SetFromJSON(json.dumps(script_dict))

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(class_name)
        launch_info.SetScriptedProcessDictionary(structured_data)
        return launch_info

    def multiplex(self, muxed_process):
        muxed_process.GetScriptedImplementation().multiplexer = (
            self.mux_process.GetScriptedImplementation()
        )
        self.mux_process.GetScriptedImplementation().multiplexed_processes[
            muxed_process.GetProcessID()
        ] = muxed_process

    def passthrough_launch(self):
        """Test that a simple passthrough wrapper functions correctly"""
        # First build the real target:
        self.assertEqual(self.dbg.GetNumTargets(), 1)
        real_target_id = 0
        real_target = self.dbg.GetTargetAtIndex(real_target_id)
        lldbutil.run_break_set_by_source_regexp(self, "Break here")
        self.assertEqual(real_target.GetNumBreakpoints(), 1)

        # Now source in the scripted module:
        script_path = os.path.join(self.getSourceDir(), self.script_file)
        self.runCmd(f"command script import '{script_path}'")

        self.mux_target = self.duplicate_target(real_target)
        self.assertTrue(self.mux_target.IsValid(), "duplicate target succeeded")

        mux_class = f"{self.script_module}.MultiplexerScriptedProcess"
        script_dict = {"driving_target_idx": real_target_id}
        mux_launch_info = self.get_launch_info(mux_class, script_dict)
        self.mux_process_listener = lldb.SBListener(
            "lldb.test.interactive-scripted-process.listener"
        )
        mux_launch_info.SetShadowListener(self.mux_process_listener)

        self.dbg.SetAsync(True)
        error = lldb.SBError()
        self.mux_process = self.mux_target.Launch(mux_launch_info, error)
        self.assertSuccess(error, "Launched multiplexer scripted process")
        self.assertTrue(self.mux_process.IsValid(), "Got a valid process")

        # Check that the real process started running
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)
        # Check that the mux process started running
        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateRunning)

        # Check that the real process stopped
        event = lldbutil.fetch_next_event(
            self, self.dbg.GetListener(), self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)
        # Check that the mux process stopped
        event = lldbutil.fetch_next_event(
            self, self.mux_process_listener, self.mux_process.GetBroadcaster()
        )
        self.assertState(lldb.SBProcess.GetStateFromEvent(event), lldb.eStateStopped)

        real_process = real_target.GetProcess()
        self.assertTrue(real_process.IsValid(), "Got a valid process")
        self.assertState(
            real_process.GetState(), lldb.eStateStopped, "Process is stopped"
        )

        # This is a passthrough, so the two processes should have the same state:
        # Check that we got the right threads:
        self.assertEqual(
            len(real_process.threads),
            len(self.mux_process.threads),
            "Same number of threads",
        )
        for id in range(len(real_process.threads)):
            real_pc = real_process.threads[id].frame[0].pc
            mux_pc = self.mux_process.threads[id].frame[0].pc
            self.assertEqual(real_pc, mux_pc, f"PC's equal for {id}")
