"""
Test lldb-dap 'stopped' events.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


@skipIfWindows  # This is flakey on Windows: llvm.org/pr24668, llvm.org/pr38373
@skipIfLinux
class TestDAP_stopped_events(lldbdap_testcase.DAPTestCaseBase):
    """
    Test validates different operations that produce 'stopped' events.
    """

    ANY_THREAD = {}

    def matches(self, a: dict, b: dict) -> bool:
        """Returns true if 'a' is a subset of 'b', otherwise false."""
        return a | b == a

    def verify_threads(self, expected_threads):
        threads_resp = self.dap_server.request_threads()
        self.assertTrue(threads_resp["success"])
        threads = threads_resp["body"]["threads"]
        self.assertGreaterEqual(
            len(threads),
            len(expected_threads),
            f"thread: {threads!r}\n expected threads: {expected_threads}",
        )

        for idx, expected_thread in enumerate(expected_threads):
            thread = threads[idx]
            self.assertTrue(
                self.matches(thread, expected_thread),
                f"Invalid thread state in {threads_resp!r} for {expected_thread!r}",
            )

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained",
    )
    @expectedFailureNetBSD
    def test_multiple_threads_same_breakpoint(self):
        """
        Test that multiple threads being stopped on the same breakpoint produces multiple 'stopped' event.
        """
        program = self.getBuildArtifact("a.out")
        launch_seq = self.build_and_launch(program)
        self.dap_server.wait_for_event(["initialized"])
        [bp] = self.set_function_breakpoints(["my_add"])
        self.verify_configuration_done()
        response = self.dap_server.receive_response(launch_seq)
        self.assertTrue(response["success"])

        events = self.dap_server.wait_for_stopped()
        if len(events) == 1:
            # we may not catch both events at the same time if we run with sanitizers
            # enabled or on slow or heavily used computer.
            events.extend(self.dap_server.wait_for_stopped())

        self.assertEqual(
            len(events),
            2,
            f"Expected two 'stopped' events, seen events: {events!r}",
        )
        for event in events:
            body = event["body"]
            self.assertEqual(body["reason"], "breakpoint")
            self.assertEqual(body["text"], "breakpoint 1.1")
            self.assertEqual(body["description"], "breakpoint 1.1")
            self.assertEqual(body["hitBreakpointIds"], [int(bp)])
            self.assertIsNotNone(body["threadId"])

        # We should have three threads, something along the lines of:
        #
        # Process 1234 stopped
        #   thread #1: tid = 0x01, 0x0a libsystem_pthread.dylib`pthread_mutex_lock + 12, queue = 'com.apple.main-thread'
        # * thread #2: tid = 0x02, 0x0b a.out`add(a=1, b=2) at main.cpp:10:32, stop reason = breakpoint 1.1
        #   thread #3: tid = 0x03, 0x0c a.out`add(a=4, b=5) at main.cpp:10:32, stop reason = breakpoint 1.1
        self.verify_threads(
            [
                self.ANY_THREAD,
                {
                    "reason": "breakpoint",
                    "text": "breakpoint 1.1",
                    "description": "breakpoint 1.1",
                },
                {
                    "reason": "breakpoint",
                    "text": "breakpoint 1.1",
                    "description": "breakpoint 1.1",
                },
            ]
        )

        self.assertEqual(
            self.dap_server.threads[1]["id"],
            self.dap_server.focused_tid,
            "Expected thread#2 to be focused",
        )

        self.continue_to_exit()

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained",
    )
    @expectedFailureNetBSD
    def test_multiple_breakpoints_same_location(self):
        """
        Test stopping at a location that reports multiple overlapping breakpoints.
        """
        program = self.getBuildArtifact("a.out")
        launch_seq = self.build_and_launch(program)
        self.dap_server.wait_for_event(["initialized"])
        line_1 = line_number("main.cpp", "breakpoint")
        [bp1] = self.set_source_breakpoints("main.cpp", [line_1])
        [bp2] = self.set_function_breakpoints(["my_add"])
        self.verify_configuration_done()
        response = self.dap_server.receive_response(launch_seq)
        self.assertTrue(response["success"])

        events = self.dap_server.wait_for_stopped()
        if len(events) == 1:
            # we may not catch both events at the same time if we run with sanitizers
            # enabled or on slow or heavily used computer.
            events.extend(self.dap_server.wait_for_stopped())

        self.assertEqual(len(events), 2, "Expected two stopped events")
        for event in events:
            body = event["body"]
            self.assertEqual(body["reason"], "breakpoint")
            self.assertEqual(body["text"], "breakpoint 1.1 2.1")
            self.assertEqual(body["description"], "breakpoint 1.1 2.1")
            self.assertEqual(body["hitBreakpointIds"], [int(bp1), int(bp2)])
            self.assertIsNotNone(body["threadId"])

        # Should return something like:
        # Process 1234 stopped
        #   thread #1: tid = 0x01, 0x0a libsystem_pthread.dylib`pthread_mutex_lock + 12, queue = 'com.apple.main-thread'
        # * thread #2: tid = 0x02, 0x0b a.out`add(a=1, b=2) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
        #   thread #3: tid = 0x03, 0x0c a.out`add(a=4, b=5) at main.cpp:10:32, stop reason = breakpoint 1.1 2.1
        self.verify_threads(
            [
                self.ANY_THREAD,
                {
                    "reason": "breakpoint",
                    "description": "breakpoint 1.1 2.1",
                    "text": "breakpoint 1.1 2.1",
                },
                {
                    "reason": "breakpoint",
                    "description": "breakpoint 1.1 2.1",
                    "text": "breakpoint 1.1 2.1",
                },
            ]
        )

        self.assertEqual(
            self.dap_server.threads[1]["id"],
            self.dap_server.focused_tid,
            "Expected thread#2 to be focused",
        )

        self.continue_to_exit()
