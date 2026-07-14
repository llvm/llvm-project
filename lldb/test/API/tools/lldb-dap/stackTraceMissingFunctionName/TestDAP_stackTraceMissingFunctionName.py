"""
Test lldb-dap stack trace response
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_stackTraceMissingFunctionName(DAPTestCaseBase):
    @skipIfWindows
    def test_missingFunctionName(self):
        """
        Test that the stack frame without a function name is given its pc in the response.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program))

        stop_event = session.verify_stopped_on_exception(after=process_event)
        frame_without_function_name = session.top_frame_from(stop_event)
        self.assertEqual(frame_without_function_name.name, "0x0000000000000000")
