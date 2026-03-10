from armetm_testcase import *
from lldbsuite.test.lldbtest import *


class TestArmETMTraceLoad(TraceArmETMTestCaseBase):
    @testSBAPIAndCommands
    def testLoadTrace(self):
        src_dir = self.getSourceDir()
        trace_bundle_path = os.path.join(src_dir, "trace", "trace.json")
        self.traceLoad(trace_bundle_path)

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertEqual(process.GetProcessID(), 0)

        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetThreadAtIndex(0).GetThreadID(), 0)

        self.assertEqual(target.GetNumModules(), 1)
        module = target.GetModuleAtIndex(0)
        path = module.GetFileSpec()
        self.assertEqual(
            path.fullpath, os.path.join(src_dir, "trace", "picow_wifi_scan.elf")
        )
        self.assertGreater(module.GetNumSections(), 0)
        self.assertEqual(module.GetSectionAtIndex(0).GetFileAddress(), 0x10000000)
