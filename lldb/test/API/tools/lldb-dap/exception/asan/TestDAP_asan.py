"""
Test that we stop at runtime instrumentation locations (asan).
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_asan(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessAddressSanitizer
    def test_asan(self):
        """
        Test that we stop at asan.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.do_continue()

        self.verify_stop_exception_info("Use of deallocated memory")
        exceptionInfo = self.get_exceptionInfo()
        self.assertEqual(exceptionInfo["breakMode"], "always")
        self.assertRegex(
            exceptionInfo["description"], r"fatal_error: heap-use-after-free"
        )
        self.assertEqual(exceptionInfo["exceptionId"], "runtime-instrumentation")
