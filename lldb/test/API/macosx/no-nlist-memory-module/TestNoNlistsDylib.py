"""
Test that we read don't read the nlist symbols for a specially marked dylib
when read from memory.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from time import sleep


class NoNlistsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipUnlessDarwin
    def test_no_nlist_symbols(self):
        self.build()

        exe = os.path.realpath(self.getBuildArtifact("a.out"))

        popen = self.spawnSubprocess(exe)
        pid = popen.pid

        self.dbg.SetAsync(False)

        m = lldb.SBModule()
        target = lldb.SBTarget()
        process = lldb.SBProcess()
        reattach_count = 0

        # Attach to the process, see if we have a memory module
        # for libno-nlists.dylib.  If not, detach, delete the
        # Target, and flush the orphaned modules from the Debugger
        # so we don't hold on to a reference of the on-disk binary.

        # If we haven't succeeded after ten attemps of attaching and
        # detaching, fail the test.
        while not m.IsValid() or m.IsFileBacked():
            if process.IsValid():
                process.Detach()
                self.dbg.DeleteTarget(target)
                self.dbg.MemoryPressureDetected()
                time.sleep(2)

            self.runCmd("process attach -p " + str(pid))
            target = self.dbg.GetSelectedTarget()
            process = target.GetProcess()
            m = target.FindModule(lldb.SBFileSpec("libno-nlists.dylib"))

            reattach_count = reattach_count + 1
            if reattach_count > 10:
                break

        self.assertTrue(process, PROCESS_IS_VALID)

        # Test that we found libno-nlists.dylib, it is a memory
        # module, and that it has no symbols.
        self.assertTrue(m.IsValid())
        self.assertFalse(m.IsFileBacked())
        self.assertEqual(m.GetNumSymbols(), 0)

        # And as a sanity check, get the main binary's module,
        # test that it is file backed and that it has more than
        # zero symbols.
        m = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertTrue(m.IsValid())
        self.assertTrue(m.IsFileBacked())
        self.assertGreater(m.GetNumSymbols(), 0)
