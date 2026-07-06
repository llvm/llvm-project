"""
Test that we stop at runtime instrumentation locations (asan).
"""

from lldbsuite.test.decorators import skipUnlessAddressSanitizer
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_asan(DAPTestCaseBase):
    @skipUnlessAddressSanitizer
    def test_asan(self):
        """
        Test that we stop at asan.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program))
        stop_event = session.verify_stopped_on_exception(
            after=process_event, expected_description="Use of deallocated memory"
        )

        thread_id = self.expect_not_none(stop_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)
        self.assertEqual(exception_info.breakMode, "always")
        description = self.expect_not_none(exception_info.description)
        self.assertRegex(description, r"fatal_error: heap-use-after-free")
        self.assertEqual(exception_info.exceptionId, "runtime-instrumentation")
