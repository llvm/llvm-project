from armetm_testcase import *
from lldbsuite.test.lldbtest import *


class TestArmETMTraceSchema(TraceArmETMTestCaseBase):
    def testSchema(self):
        self.expect("trace schema arm-etm", substrs=["triple", "threads", "etmTrace"])

    def testInvalidPluginSchema(self):
        self.expect(
            "trace schema invalid-plugin",
            error=True,
            substrs=[
                'error: no trace plug-in matches the specified type: "invalid-plugin"'
            ],
        )

    def testAllSchemas(self):
        self.expect(
            "trace schema all",
            substrs=[
                """{
  "type": "arm-etm",
  """
            ],
        )
