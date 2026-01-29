"""
Test that we stop at runtime instrumentation locations.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_runtime_instruments(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessUndefinedBehaviorSanitizer
    def test_ubsan(self):
        """
        Test that we stop at ubsan.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.do_continue()

        self.verify_stop_exception_info("Out of bounds index")
        exceptionInfo = self.get_exceptionInfo()
        self.assertEqual(exceptionInfo["breakMode"], "always")
        self.assertRegex(exceptionInfo["description"], r"Out of bounds index")
        self.assertEqual(exceptionInfo["exceptionId"], "runtime-instrumentation")
        self.assertIn("main.c", exceptionInfo["details"]["stackTrace"])
