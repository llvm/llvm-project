import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestDisableLanguageUnwinder(lldbtest.TestBase):

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test async unwind"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', src)

        self.assertIn("syncFunc", thread.GetFrameAtIndex(0).GetFunctionName())
        self.assertIn("callSyncFunc", thread.GetFrameAtIndex(1).GetFunctionName())
        self.assertIn("main", thread.GetFrameAtIndex(2).GetFunctionName())

        self.runCmd("settings set target.process.disable-language-runtime-unwindplans true")

        self.assertIn("syncFunc", thread.GetFrameAtIndex(0).GetFunctionName())
        self.assertIn("callSyncFunc", thread.GetFrameAtIndex(1).GetFunctionName())
        self.assertNotIn("main", thread.GetFrameAtIndex(2).GetFunctionName())
