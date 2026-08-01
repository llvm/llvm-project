"""
Test lldb-dap variables/stackTrace request for optimized code
"""

from lldbsuite.test.decorators import skipIfAsan, skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_optimized(DAPTestCaseBase):
    @skipIfWindows
    def test_stack_frame_name(self):
        """Test optimized frame has special name suffix."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        source = "main.cpp"
        breakpoint_line = line_number(source, "// breakpoint 1")
        with session.configure(LaunchArgs(program)) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint_line])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_ids, after=ctx.process_event
        )
        frames = session.thread_context_from(stop_event).frames()

        leaf_frame = frames[0].frame
        self.assertTrue(leaf_frame.name.endswith(" [opt]"))
        parent_frame = frames[1].frame
        self.assertTrue(parent_frame.name.endswith(" [opt]"))

    @skipIfAsan  # On ASAN builds this test intermittently fails https://github.com/llvm/llvm-project/issues/111061
    @skipIfWindows
    def test_optimized_variable(self):
        """Test optimized variable value contains error."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"
        breakpoint_line = line_number(source, "// breakpoint 2")
        with session.configure(LaunchArgs(program)) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint_line])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_ids, after=ctx.process_event
        )
        optimized_variable = session.top_frame_from(stop_event).locals["argc"]
        value = optimized_variable.value

        self.assertTrue(
            value.startswith("<error:"), f"expect error for value: '{value}'"
        )
        self.assertTrue(
            ("could not evaluate DW_OP_entry_value: no parent function" in value)
            or ("variable not available" in value),
            f"{value=}",
        )
        session.continue_to_exit()
