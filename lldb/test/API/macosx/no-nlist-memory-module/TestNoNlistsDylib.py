"""
Test that we read don't read the nlist symbols for a specially marked dylib
when read from memory.
"""

import os
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

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(
            self, "pid_file_%d" % (int(time.time()))
        )
        self.addTearDownHook(
            lambda: self.run_platform_command("rm %s" % (pid_file_path))
        )

        # Spawn a new process
        popen = self.spawnSubprocess(exe, [pid_file_path])

        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        os.unlink(self.getBuildArtifact("libno-nlists.dylib"))
        os.unlink(self.getBuildArtifact("libhas-nlists.dylib"))

        self.runCmd("process attach -p " + str(pid))
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        m_no_nlist = target.FindModule(lldb.SBFileSpec("libno-nlists.dylib"))
        m_has_nlist = target.FindModule(lldb.SBFileSpec("libhas-nlists.dylib"))

        self.assertTrue(process, PROCESS_IS_VALID)

        if self.TraceOn():
            self.runCmd("image list")
            self.runCmd("target modules dump symtab libno-nlists.dylib")
            self.runCmd("target modules dump symtab libhas-nlists.dylib")

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
