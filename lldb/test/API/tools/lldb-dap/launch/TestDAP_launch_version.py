from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_version(DAPTestCaseBase):
    """
    Tests that "initialize" response contains the "version" string the same
    as the one returned by "version" command.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        process_event = session.launch(LaunchArgs(program=program, stopOnEntry=True))
        session.verify_stopped_on_entry(after=process_event)

        version_eval_output = session.evaluate("`version", context="repl").result
        version_string = self.expect_not_none(session.capabilities().lldb_version)

        self.assertEqual(
            version_eval_output.splitlines(),
            version_string.splitlines(),
            "version string does not match",
        )
