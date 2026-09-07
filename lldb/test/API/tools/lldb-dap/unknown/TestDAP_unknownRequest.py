"""
Test lldb-dap unknown request.
"""

from dataclasses import dataclass
from typing import Optional

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import EmptyBodyResponse, LaunchArgs


@dataclass(frozen=True)
class UnknownArgs:
    foo: Optional[str] = None
    id: Optional[int] = None

    command_ = "unknown"
    response_class_ = EmptyBodyResponse


class TestDAP_unknown_request(DAPTestCaseBase):
    """
    Tests handling of unknown request.
    """

    def test(self):
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        process_event = session.launch(LaunchArgs(program, stopOnEntry=True))
        session.verify_stopped_on_entry(after=process_event)

        # Test without arguments.
        unknown_args = UnknownArgs()
        response = session.send_request(unknown_args).error()
        resp_body = self.expect_not_none(response.body)
        resp_error = self.expect_not_none(resp_body.error)
        self.assertEqual(resp_error.format, "unknown request")

        # Test with arguments.
        unknown_args = UnknownArgs(foo="bar", id=42)
        response = session.send_request(unknown_args).error()
        resp_body = self.expect_not_none(response.body)
        resp_error = self.expect_not_none(resp_body.error)
        self.assertEqual(resp_error.format, "unknown request")

        session.continue_to_exit()
