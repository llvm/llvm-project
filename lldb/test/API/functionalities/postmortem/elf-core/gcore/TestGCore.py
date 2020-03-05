"""
Test signal reporting when debugging with linux core files.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class GCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    _i386_pid = 5586
    _x86_64_pid = 5669

    @skipIf(oslist=['windows'])
    @skipIf(triple='^mips')
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("linux-i386", self._i386_pid)

    @skipIf(oslist=['windows'])
    @skipIf(triple='^mips')
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("linux-x86_64", self._x86_64_pid)

    def do_test(self, filename, pid):
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(filename + ".core")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 3)
        self.assertEqual(process.GetProcessID(), pid)

        for thread in process:
            reason = thread.GetStopReason()
            self.assertEqual(reason, lldb.eStopReasonSignal)
            signal = thread.GetStopReasonDataAtIndex(1)
            # Check we got signal 19 (SIGSTOP)
            self.assertEqual(signal, 19)

        self.dbg.DeleteTarget(target)
