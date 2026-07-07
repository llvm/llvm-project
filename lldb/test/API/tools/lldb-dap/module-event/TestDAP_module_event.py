from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs, StoppedReason
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_module_event(DAPTestCaseBase):
    @skipIfWindows
    def test_module_event(self):
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")

        source = "main.cpp"
        bp1_line = line_number(source, "// breakpoint 1")
        bp2_line = line_number(source, "// breakpoint 2")
        bp3_line = line_number(source, "// breakpoint 3")

        with session.configure(LaunchArgs(program=program)) as ctx:
            [bp1, bp2, bp3] = session.resolve_source_breakpoints(
                source, [bp1_line, bp2_line, bp3_line]
            )
        # Wait for the breakpoint before dlopen.
        before_dlopen_event = session.verify_stopped_on_breakpoint(
            bp1, after=ctx.process_event
        )

        # Continue to the second breakpoint, before the dlclose.
        session.continue_to_breakpoint(bp2)

        # Make sure we got a module event for libother.
        new_module_event = session.verify_next_module_event(after=before_dlopen_event)
        module_id = new_module_event.body.module.id
        self.assertEqual(new_module_event.body.reason, "new")
        self.assertIn("libother", new_module_event.body.module.name)

        # Continue to the third breakpoint, after the dlclose.
        session.continue_to_breakpoint(bp3)

        # Make sure we got a module event for libother.
        removed_module_event = session.verify_next_module_event(after=new_module_event)
        reason = removed_module_event.body.reason
        self.assertEqual(reason, "removed")
        self.assertEqual(removed_module_event.body.module.id, module_id)

        # The removed module event should omit everything but the module id and name
        # as they are required fields.
        removed_module = removed_module_event.body.module
        self.assertIsNotNone(removed_module.id)
        self.assertIsNotNone(removed_module.name)
        self.assertEqual(removed_module.name, "", "expects empty name.")

        session.continue_to_exit()
