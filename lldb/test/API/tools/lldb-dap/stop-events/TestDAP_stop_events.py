"""
Test lldb-dap stop events.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_stop_events(lldbdap_testcase.DAPTestCaseBase):
    """
    Test validates different operations that produce 'stopped' events.
    """

    def evaluate(self, command: str) -> str:
        result = self.dap_server.request_evaluate(command, context="repl")
        self.assertTrue(result["success"])
        return result["body"]["result"]

    def test_multiple_threads_sample_breakpoint(self):
        """
        Test that multiple threads being stopped on the same breakpoint only produces a single 'stopped' event.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        line_1 = line_number("main.cpp", "breakpoint 1")
        [bp1] = self.set_source_breakpoints("main.cpp", [line_1])

        events = self.continue_to_next_stop()
        self.assertEqual(len(events), 1, "Expected a single stopped event")
        body = events[0]["body"]
        self.assertEqual(body["reason"], "breakpoint")
        self.assertEqual(body["text"], "breakpoint 1.1")
        self.assertEqual(body["description"], "breakpoint 1.1")
        self.assertEqual(body["hitBreakpointIds"], [int(bp1)])
        self.assertEqual(body["allThreadsStopped"], True)
        self.assertNotIn("preserveFocusHint", body)
        self.assertIsNotNone(body["threadId"])

        # Should return something like:
        # Process 1234 stopped
        #   thread #1: tid = 0x01, 0x0a libsystem_pthread.dylib`pthread_mutex_lock + 12, queue = 'com.apple.main-thread'
        # * thread #2: tid = 0x02, 0x0b a.out`add(a=1, b=2) at main.cpp:10:32, stop reason = breakpoint 1.1
        #   thread #3: tid = 0x03, 0x0c a.out`add(a=4, b=5) at main.cpp:10:32, stop reason = breakpoint 1.1
        result = self.evaluate("thread list")

        # Ensure we have 2 threads stopped at the same breakpoint.
        threads_with_stop_reason = [
            l for l in result.split("\n") if "stop reason = breakpoint" in l
        ]
        self.assertTrue(
            len(threads_with_stop_reason) == 2,
            f"Failed to stop at the same breakpoint: {result}",
        )

        self.continue_to_exit()

    def test_multiple_breakpoints_same_location(self):
        """
        Test stopping at a location that reports multiple overlapping breakpoints.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        line_1 = line_number("main.cpp", "breakpoint 1")
        [bp1] = self.set_source_breakpoints("main.cpp", [line_1])
        [bp2] = self.set_function_breakpoints(["my_add"])

        events = self.continue_to_next_stop()
        self.assertEqual(len(events), 1, "Expected a single stopped event")
        body = events[0]["body"]
        self.assertEqual(body["reason"], "breakpoint")
        self.assertEqual(body["text"], "breakpoint 1.1 2.1")
        self.assertEqual(body["description"], "breakpoint 1.1 2.1")
        self.assertEqual(body["hitBreakpointIds"], [int(bp1), int(bp2)])
        self.assertEqual(body["allThreadsStopped"], True)
        self.assertNotIn("preserveFocusHint", body)
        self.assertIsNotNone(body["threadId"])

        # Should return something like:
        # Process 1234 stopped
        #   thread #1: tid = 0x01, 0x0a libsystem_pthread.dylib`pthread_mutex_lock + 12, queue = 'com.apple.main-thread'
        # * thread #2: tid = 0x02, 0x0b a.out`add(a=1, b=2) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
        #   thread #3: tid = 0x03, 0x0c a.out`add(a=4, b=5) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
        result = self.evaluate("thread list")

        # Ensure we have 2 threads at the same location with overlapping breakpoints.
        threads_with_stop_reason = [
            l for l in result.split("\n") if "stop reason = breakpoint" in l
        ]
        self.assertTrue(
            len(threads_with_stop_reason) == 2,
            f"Failed to stop at the same breakpoint: {result}",
        )
        self.continue_to_exit()
