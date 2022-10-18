from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestTraceDumpInfo(TraceIntelPTTestCaseBase):
    def testDumpFunctionCalls(self):
      self.expect("trace load -v " +
        os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"))

      self.expect("thread trace dump function-calls 2",
        error=True, substrs=['error: no thread with index: "2"'])

      self.expect("thread trace dump function-calls 1 -j",
        substrs=['json = true, pretty_json = false, file = false, thread = 3842849'])

      self.expect("thread trace dump function-calls 1 -F /tmp -J",
        substrs=['false, pretty_json = true, file = true, thread = 3842849'])
