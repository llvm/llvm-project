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


    def testLoadInvalidTraces(self):
        src_dir = self.getSourceDir()
        # We test first an invalid type
        trace_definition_file = os.path.join(src_dir, "intelpt-trace", "trace_bad.json")
        self.expect("trace load -v " + trace_definition_file, error=True,
          substrs=['error: JSON value is expected to be "object"', "Value", "123", "Schema"])

        # Now we test a missing field
        trace_definition_file2 = os.path.join(src_dir, "intelpt-trace", "trace_bad2.json")
        self.expect("trace load -v " + trace_definition_file2, error=True,
            substrs=['error: JSON object is missing the "triple" field.', "Value", "pid", "12345", "Schema"])

        # The following wrong schema will have a valid target and an invalid one. In the case of failure,
        # no targets should be created.
        self.assertEqual(self.dbg.GetNumTargets(), 0)
        trace_definition_file2 = os.path.join(src_dir, "intelpt-trace", "trace_bad3.json")
        self.expect("trace load -v " + trace_definition_file2, error=True,
            substrs=['error: JSON object is missing the "pid" field.'])
        self.assertEqual(self.dbg.GetNumTargets(), 0)
