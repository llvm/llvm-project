"""
Test lldb-dap launch request.
"""

import lldbdap_testcase


class TestDAP_launch_stopOnEntry(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the default launch of a simple program that stops at the
    entry point instead of continuing.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.dap_server.request_configurationDone()
        self.dap_server.wait_for_stopped()
        self.assertTrue(
            len(self.dap_server.thread_stop_reasons) > 0,
            "expected stopped event during launch",
        )
        for _, body in self.dap_server.thread_stop_reasons.items():
            if "reason" in body:
                reason = body["reason"]
                self.assertNotEqual(
                    reason, "breakpoint", 'verify stop isn\'t "main" breakpoint'
                )
