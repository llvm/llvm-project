import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceLoad(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def testSchema(self):
        self.expect("trace schema intel-pt", substrs=["triple", "threads", "traceBuffer"])

    def testInvalidPluginSchema(self):
        self.expect("trace schema invalid-plugin", error=True,
            substrs=['error: no trace plug-in matches the specified type: "invalid-plugin"'])

    def testAllSchemas(self):
        self.expect("trace schema all", substrs=['''{
  "type": "intel-pt",
  "cpuInfo": {
    // CPU information gotten from, for example, /proc/cpuinfo.

    "vendor": "GenuineIntel" | "unknown",
    "family": integer,
    "model": integer,
    "stepping": integer
  },'''])
