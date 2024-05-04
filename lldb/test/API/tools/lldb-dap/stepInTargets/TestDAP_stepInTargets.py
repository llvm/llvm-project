"""
Test lldb-dap stepInTargets request
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase
from lldbsuite.test import lldbutil


class TestDAP_stepInTargets(lldbdap_testcase.DAPTestCaseBase):
    @skipIf(
        archs=no_match(["x86_64"])
    )  # InstructionControlFlowKind for ARM is not supported yet.
    def test_basic(self):
        """
        Tests the basic stepping in targets with directly calls.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"

        breakpoint_line = line_number(source, "// set breakpoint here")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)

        threads = self.dap_server.get_threads()
        self.assertEqual(len(threads), 1, "expect one thread")
        tid = threads[0]["id"]

        leaf_frame = self.dap_server.get_stackFrame()
        self.assertIsNotNone(leaf_frame, "expect a leaf frame")

        # Request all step in targets list and verify the response.
        step_in_targets_response = self.dap_server.request_stepInTargets(
            leaf_frame["id"]
        )
        self.assertEqual(step_in_targets_response["success"], True, "expect success")
        self.assertIn(
            "body", step_in_targets_response, "expect body field in response body"
        )
        self.assertIn(
            "targets",
            step_in_targets_response["body"],
            "expect targets field in response body",
        )

        step_in_targets = step_in_targets_response["body"]["targets"]
        self.assertEqual(len(step_in_targets), 3, "expect 3 step in targets")

        # Verify the target names are correct.
        self.assertEqual(step_in_targets[0]["label"], "bar()", "expect bar()")
        self.assertEqual(step_in_targets[1]["label"], "bar2()", "expect bar2()")
        self.assertEqual(
            step_in_targets[2]["label"], "foo(int, int)", "expect foo(int, int)"
        )

        # Choose to step into second target and verify that we are in bar2()
        self.stepIn(threadId=tid, targetId=step_in_targets[1]["id"], waitForStop=True)
        leaf_frame = self.dap_server.get_stackFrame()
        self.assertIsNotNone(leaf_frame, "expect a leaf frame")
        self.assertEqual(leaf_frame["name"], "bar2()")
