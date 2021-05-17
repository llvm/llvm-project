import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceLoad(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        if 'intel-pt' not in configuration.enabled_plugins:
            self.skipTest("The intel-pt test plugin is not enabled")


    def testLoadTrace(self):
        src_dir = self.getSourceDir()
        trace_definition_file = os.path.join(src_dir, "intelpt-trace", "trace.json")
        self.expect("trace load -v " + trace_definition_file, substrs=["intel-pt"])

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertEqual(process.GetProcessID(), 1234)

        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetThreadAtIndex(0).GetThreadID(), 3842849)

        self.assertEqual(target.GetNumModules(), 1)
        module = target.GetModuleAtIndex(0)
        path = module.GetFileSpec()
        self.assertEqual(path.fullpath, os.path.join(src_dir, "intelpt-trace", "a.out"))
        self.assertGreater(module.GetNumSections(), 0)
        self.assertEqual(module.GetSectionAtIndex(0).GetFileAddress(), 0x400000)

        self.assertEqual("6AA9A4E2-6F28-2F33-377D-59FECE874C71-5B41261A", module.GetUUIDString())

        # check that the Process and Thread objects were created correctly
        self.expect("thread info", substrs=["tid = 3842849"])
        self.expect("thread list", substrs=["Process 1234 stopped", "tid = 3842849"])


    def testLoadInvalidTraces(self):
        src_dir = self.getSourceDir()
        # We test first an invalid type
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad.json"), error=True,
          substrs=['''error: expected object at traceSession.processes[0]

Context:
{
  "processes": [
    /* error: expected object */
    123
  ],
  "trace": { ... }
}

Schema:
{
  "trace": {
    "type": "intel-pt",
    "cpuInfo": {
      "vendor": "intel" | "unknown",
      "family": integer,
      "model": integer,
      "stepping": integer
    }
  },'''])

        # Now we test a missing field in the global session file
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad2.json"), error=True,
            substrs=['error: missing value at traceSession.processes[1].triple', "Context", "Schema"])

        # Now we test a missing field in the intel-pt settings
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad4.json"), error=True,
            substrs=['''error: missing value at traceSession.trace.cpuInfo.family

Context:
{
  "processes": [],
  "trace": {
    "cpuInfo": /* error: missing value */ {
      "model": 79,
      "stepping": 1,
      "vendor": "intel"
    },
    "type": "intel-pt"
  }
}''', "Schema"])

        # Now we test an incorrect load address in the intel-pt settings
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad5.json"), error=True,
            substrs=['error: expected numeric string at traceSession.processes[0].modules[0].loadAddress',
                     '"loadAddress": /* error: expected numeric string */ 400000,', "Schema"])

        # The following wrong schema will have a valid target and an invalid one. In the case of failure,
        # no targets should be created.
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        self.expect("trace load -v " + os.path.join(src_dir, "intelpt-trace", "trace_bad3.json"), error=True,
            substrs=['error: missing value at traceSession.processes[1].pid'])
        self.assertEqual(self.dbg.GetNumTargets(), 0)
