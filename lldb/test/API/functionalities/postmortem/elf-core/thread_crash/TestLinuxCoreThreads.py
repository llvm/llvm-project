"""
Test signal reporting when debugging with linux core files.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LinuxCoreThreadsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    _i386_pid = 5193
    _x86_64_pid = 5222

    # Thread id for the failing thread.
    _i386_tid = 5195
    _x86_64_tid = 5250

    @skipIf(oslist=['windows'])
    @skipIf(triple='^mips')
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("linux-i386", self._i386_pid, self._i386_tid)

    @skipIf(oslist=['windows'])
    @skipIf(triple='^mips')
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_tid)

    def do_test(self, filename, pid, tid):
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(filename + ".core")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 3)
        self.assertEqual(process.GetProcessID(), pid)

        for thread in process:
            # Verify that if we try to read memory from a PT_LOAD that has
            # p_filesz of zero that we don't get bytes from the next section
            # that actually did have bytes. The addresses below were found by
            # dumping the program headers of linux-i386.core and
            # linux-x86_64.core and verifying that they had a p_filesz of zero.
            mem_err = lldb.SBError()
            if process.GetAddressByteSize() == 4:
                bytes_read = process.ReadMemory(0x8048000, 4, mem_err)
            else:
                bytes_read = process.ReadMemory(0x400000, 4, mem_err)
            self.assertEqual(bytes_read, None)
            reason = thread.GetStopReason()
            if( thread.GetThreadID() == tid ):
                self.assertEqual(reason, lldb.eStopReasonSignal)
                signal = thread.GetStopReasonDataAtIndex(1)
                # Check we got signal 4 (SIGILL)
                self.assertEqual(signal, 4)
            else:
                signal = thread.GetStopReasonDataAtIndex(1)
                # Check we got no signal on the other threads
                self.assertEqual(signal, 0)

        self.dbg.DeleteTarget(target)
