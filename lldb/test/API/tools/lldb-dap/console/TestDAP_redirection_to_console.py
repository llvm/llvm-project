"""Test that lldb-dap keeps stdout/stderr redirection working even when the
inferior's output is routed back through the debug console."""

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.utils import DebugAdapterOptions


class TestDAP_redirection_to_console(DAPTestCaseBase):
    USE_DEFAULT_DEBUG_ADAPTER = False

    def test(self):
        """
        Without proper stderr and stdout redirection, the following code would throw an
        exception, like the following:

            Exception: unexpected malformed message from lldb-dap
        """
        self.build()
        program = self.getBuildArtifact("a.out")
        adapter = self.create_stdio_debug_adapter(
            DebugAdapterOptions(
                env={"LLDB_DAP_TEST_STDOUT_STDERR_REDIRECTION": ""},
            )
        )
        session = self.create_session(adapter)

        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program)) as ctx:
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint1_line])

        stop = session.verify_stopped_on_breakpoint(bp_id, after=ctx.process_event)
        thread_ctx = session.thread_context_from(stop)
        _, second_frame, *_ = thread_ctx.frames()
        local_names = [var.name for var in second_frame.locals.variables()]
        self.assertIn("argc", local_names)
