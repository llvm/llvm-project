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

        m_no_nlist = lldb.SBModule()
        m_has_nlist = lldb.SBModule()
        target = lldb.SBTarget()
        process = lldb.SBProcess()
        reattach_count = 0

        # Attach to the process, see if we have a memory module
        # for libno-nlists.dylib and libhas-nlists.dylib.
        # If not, detach, delete the Target, and flush the orphaned
        # modules from the Debugger so we don't hold on to a reference
        # of the on-disk binary.

        # If we haven't succeeded after ten attemps of attaching and
        # detaching, fail the test.
        while (not m_no_nlist.IsValid() or m_no_nlist.IsFileBacked()) and (
            not m_has_nlist.IsValid() or m_has_nlist.IsFileBacked()
        ):
            if process.IsValid():
                process.Detach()
                self.dbg.DeleteTarget(target)
                self.dbg.MemoryPressureDetected()
                time.sleep(2)

            self.runCmd("process attach -p " + str(pid))
            target = self.dbg.GetSelectedTarget()
            process = target.GetProcess()
            m_no_nlist = target.FindModule(lldb.SBFileSpec("libno-nlists.dylib"))
            m_has_nlist = target.FindModule(lldb.SBFileSpec("libhas-nlists.dylib"))

            reattach_count = reattach_count + 1
            if reattach_count > 10:
                break

        self.assertTrue(process, PROCESS_IS_VALID)

        # Test that we found libno-nlists.dylib, it is a memory
        # module, and that it has no symbols.
        self.assertTrue(m_no_nlist.IsValid())
        self.assertFalse(m_no_nlist.IsFileBacked())
        self.assertEqual(m_no_nlist.GetNumSymbols(), 0)

        # Test that we found libhas-nlists.dylib, it is a memory
        # module, and that it has more than zero symbols.
        self.assertTrue(m_has_nlist.IsValid())
        self.assertFalse(m_has_nlist.IsFileBacked())
        self.assertGreater(m_has_nlist.GetNumSymbols(), 0)

        # And as a sanity check, get the main binary's module,
        # test that it is file backed and that it has more than
        # zero symbols.
        m_exe = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertTrue(m_exe.IsValid())
        self.assertTrue(m_exe.IsFileBacked())
        self.assertGreater(m_exe.GetNumSymbols(), 0)
