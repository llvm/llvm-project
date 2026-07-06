"""
Test exception behavior in DAP with c++ throw.
"""

from lldbsuite.test.decorators import skipIfWasm, skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


@skipIfWasm  # wasm inferiors are built with -fno-exceptions.
class TestDAP_exception_cpp(DAPTestCaseBase):
    @skipIfWindows
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped
        event.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program=program))

        stopped_event = session.verify_stopped_on_exception(
            expected_description="signal SIGABRT", after=process_event
        )

        thread_id = self.expect_not_none(stopped_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)

        self.assertEqual(exception_info.breakMode, "always")
        description = self.expect_not_none(exception_info.description)
        self.assertIn("signal SIGABRT", description)
        self.assertEqual(exception_info.exceptionId, "signal")
        self.assertIsNotNone(exception_info.details)

    @skipIfWindows  # cpp exception breakpoint is not supported.
    def test_break_on_throw_and_catch(self):
        """Test thrown and caught cpp exception breakpoint works."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program, args=["1"])) as ctx:
            # Check we advertise the c++ throw and catch exception filters.
            resp = ctx.init_response
            bp_filters = self.expect_not_none(resp.body.exceptionBreakpointFilters)

            init_filters = [ft.filter for ft in bp_filters]
            cpp_filters = ["cpp_throw", "cpp_catch"]
            for filter in cpp_filters:
                self.assertIn(filter, init_filters, "expect cpp_filters is advertised.")

            session.set_exception_breakpoints(filters=cpp_filters)

        stop_event = session.verify_stopped_on_exception(after=ctx.process_event)
        thread_ctx = session.thread_context_from(stop_event)

        def verify_stack_trace_contains(function: str, line: int):
            frames = [ctx.frame for ctx in thread_ctx.frames()]
            for frame in frames:
                if frame.name.startswith(function):
                    frame_line = self.expect_not_none(frame.line)
                    self.assertEqual(frame_line, line)
                    return

            msg = f"expected {function=} {line=} in stacktrace:"
            for i, frame in enumerate(frames):
                msg += f"\n [{i}] {frame.name}:{frame.line}"
            self.fail(msg)

        source = self.getSourcePath("main.cpp")
        [throw_filter, caught_filter] = cpp_filters

        # Check the thrown exception.
        thrown_line = line_number(source, "// thrown_exception")
        verify_stack_trace_contains("throw_some_string", thrown_line)
        exc_info = session.get_exception_info(thread_ctx.thread_id)
        self.assertEqual(throw_filter, exc_info.exceptionId)

        # Check the caught exception.
        session.continue_to_exception_breakpoint()
        caught_line = line_number(source, "// caught_exception")
        verify_stack_trace_contains("main", caught_line)
        exc_info = session.get_exception_info(thread_ctx.thread_id)
        self.assertEqual(caught_filter, exc_info.exceptionId)

        session.continue_to_exit()
