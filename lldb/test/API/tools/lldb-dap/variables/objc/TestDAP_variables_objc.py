"""
Test 'variables' requests for obj-c types.
"""

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_variables_objc(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessDarwin
    def test_objc_description(self):
        """Test that we can get the description of an Objective-C object."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
        )
        source = "main.m"
        breakpoint_ids = self.set_source_breakpoints(
            source, [line_number(source, "// breakpoint")]
        )
        self.continue_to_breakpoints(breakpoint_ids)

        greeter_var = self.dap_server.get_local_variable(name="greeter")
        self.assertIsNotNone(greeter_var, "greeter variable should not be None")
        self.assertEqual(greeter_var["type"], "Greeter *")
        self.assertEqual(greeter_var["evaluateName"], "greeter")
        self.assertRegexpMatches(
            greeter_var["value"], r"<Greeter 0x[0-9A-Fa-f]+ name=Bob debugDescription>"
        )
        self.continue_to_exit()
