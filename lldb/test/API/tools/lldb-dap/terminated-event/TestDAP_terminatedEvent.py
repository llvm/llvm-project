"""
Test lldb-dap terminated event
"""

import json

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_terminatedEvent(DAPTestCaseBase):
    @skipIfWindows
    def test_terminated_event(self):
        """
        Terminated Event
        Now contains the statistics of a debug session:
        metatdata:
            totalDebugInfoByteSize > 0
            totalDebugInfoEnabled > 0
            totalModuleCountHasDebugInfo > 0
            ...
        targetInfo:
            totalBreakpointResolveTime > 0
        breakpoints:
            recognize function breakpoint
            recognize source line breakpoint
        It should contain the breakpoints info: function bp & source line bp
        """
        program = self.getBuildArtifact("a.out.stripped")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.cpp")
        main_bp_line = line_number(source, "// main breakpoint 1")

        with session.configure(LaunchArgs(program=program)) as ctx:
            # This breakpoint will be resolved only when the libfoo module is loaded.
            func_response = session.set_function_breakpoints(["foo"])
            breakpoints = func_response.body.breakpoints
            breakpoints.extend(
                session.set_source_breakpoints(source, [main_bp_line]).body.breakpoints
            )
        breakpoint_ids = session.breakpoints_to_ids(breakpoints)
        self.assertEqual(len(breakpoint_ids), 2, "expect one breakpoint")
        last_bp_event = session.wait_until_any_breakpoint_hit(
            breakpoint_ids, after=ctx.process_event
        )
        session.continue_to_exit()

        terminated = session.wait_for_terminated_event(after=last_bp_event)
        statistics = self.expect_not_none(terminated.body).lldb_statistics

        self.assertGreater(statistics["totalDebugInfoByteSize"], 0)
        self.assertGreater(statistics["totalDebugInfoEnabled"], 0)
        self.assertGreater(statistics["totalModuleCountHasDebugInfo"], 0)

        self.assertIsNotNone(statistics["memory"])
        self.assertNotIn("modules", statistics.keys())

        # lldb-dap debugs one target at a time.
        target = json.loads(statistics["targets"])[0]
        self.assertGreater(target["totalBreakpointResolveTime"], 0)

        breakpoints = target["breakpoints"]
        self.assertIn(
            "foo",
            breakpoints[0]["details"]["Breakpoint"]["BKPTResolver"]["Options"][
                "SymbolNames"
            ],
            "foo is a symbol breakpoint",
        )
        self.assertTrue(
            breakpoints[1]["details"]["Breakpoint"]["BKPTResolver"]["Options"][
                "FileName"
            ].endswith("main.cpp"),
            "target has source line breakpoint in main.cpp",
        )
