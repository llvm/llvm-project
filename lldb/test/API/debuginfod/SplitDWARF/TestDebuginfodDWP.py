"""
Test support for the DebugInfoD network symbol acquisition protocol.
"""
import os
import shutil
import tempfile
import struct

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


def getUUID(aoutuuid):
    """
    Pull the 20 byte UUID out of the .note.gnu.build-id section that was dumped
    to a file already, as part of the build.
    """
    with open(aoutuuid, "rb") as f:
        data = f.read(36)
        if len(data) != 36:
            return None
        header = struct.unpack_from("<4I", data)
        if len(header) != 4:
            return None
        # 4 element 'prefix', 20 bytes of uuid, 3 byte long string: 'GNU':
        if header[0] != 4 or header[1] != 20 or header[2] != 3 or header[3] != 0x554e47:
            return None
        return data[16:].hex()

"""
Test support for the DebugInfoD network symbol acquisition protocol.
This file is for split-dwarf (dwp) scenarios.

1 - A split binary target with it's corresponding DWP file
2 - A stripped, split binary target with an unstripped binary and a DWP file
3 - A stripped, split binary target with an --only-keep-debug symbols file and a DWP file
"""
class DebugInfodDWPTests(TestBase):
    # No need to try every flavor of debug inf.
    NO_DEBUG_INFO_TESTCASE = True

    def test_normal_stripped(self):
        """
        Validate behavior with a stripped binary, no symbols or symbol locator.
        """
        self.config_test(["a.out"])
        self.try_breakpoint(False)

    def test_normal_stripped_split_with_dwp(self):
        """
        Validate behavior with symbols, but no symbol locator.
        """
        self.config_test(["a.out", "a.out.debug", "a.out.dwp"])
        self.try_breakpoint(True)

    def test_normal_stripped_only_dwp(self):
        """
        Validate behavior *with* dwp symbols only, but missing other symbols,
        but no symbol locator. This shouldn't work: without the other symbols
        DWO's appear mostly useless.
        """
        self.config_test(["a.out", "a.out.dwp"])
        self.try_breakpoint(False)

    def test_debuginfod_dwp_from_service(self):
        """
        Test behavior with the unstripped binary, and DWP from the service.
        """
        self.config_test(["a.out.debug"], "a.out.dwp")
        self.try_breakpoint(True)

    def test_debuginfod_both_symfiles_from_service(self):
        """
        Test behavior with a stripped binary, with the unstripped binary and
        dwp symbols from Debuginfod.
        """
        self.config_test(["a.out"], "a.out.dwp", "a.out.full")
        self.try_breakpoint(True)

    def test_debuginfod_both_okd_symfiles_from_service(self):
        """
        Test behavior with both the only-keep-debug symbols and the dwp symbols
        from Debuginfod.
        """
        self.config_test(["a.out"], "a.out.dwp", "a.out.debug")
        self.try_breakpoint(True)

    def try_breakpoint(self, should_have_loc):
        """
        This function creates a target from self.aout, sets a function-name
        breakpoint, and checks to see if we have a file/line location,
        as a way to validate that the symbols have been loaded.
        should_have_loc specifies if we're testing that symbols have or
        haven't been loaded.
        """
        target = self.dbg.CreateTarget(self.aout)
        self.assertTrue(target and target.IsValid(), "Target is valid")

        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)

        loc = bp.GetLocationAtIndex(0)
        self.assertTrue(loc and loc.IsValid(), "Location is valid")
        addr = loc.GetAddress()
        self.assertTrue(addr and addr.IsValid(), "Loc address is valid")
        line_entry = addr.GetLineEntry()
        self.assertEqual(should_have_loc, line_entry != None and line_entry.IsValid(), "Loc line entry is valid")
        if should_have_loc:
            self.assertEqual(line_entry.GetLine(), 4)
            self.assertEqual(line_entry.GetFileSpec().GetFilename(), self.main_source_file.GetFilename())
        self.dbg.DeleteTarget(target)
        shutil.rmtree(self.tmp_dir)

    def config_test(self, local_files, debuginfo = None, executable = None):
        """
        Set up a test with local_files[] copied to a different location
        so that we control which files are, or are not, found in the file system.
        Also, create a stand-alone file-system 'hosted' debuginfod server with the
        provided debuginfo and executable files (if they exist)

        Make the filesystem look like:

        /tmp/<tmpdir>/test/[local_files]

        /tmp/<tmpdir>/cache (for lldb to use as a temp cache)

        /tmp/<tmpdir>/buildid/<uuid>/executable -> <executable>
        /tmp/<tmpdir>/buildid/<uuid>/debuginfo -> <debuginfo>
        Returns the /tmp/<tmpdir> path
        """

        self.build()

        uuid = getUUID(self.getBuildArtifact("a.out.uuid"))

        self.main_source_file = lldb.SBFileSpec("main.c")
        self.tmp_dir = tempfile.mkdtemp()
        self.test_dir = os.path.join(self.tmp_dir, "test")
        os.makedirs(self.test_dir)

        self.aout = ""
        # Copy the files used by the test:
        for f in local_files:
            shutil.copy(self.getBuildArtifact(f), self.test_dir)
            if (self.aout == ""):
                self.aout = os.path.join(self.test_dir, f)

        use_debuginfod = debuginfo != None or executable != None

        # Populated the 'file://... mocked' Debuginfod server:
        if use_debuginfod:
            os.makedirs(os.path.join(self.tmp_dir, "cache"))
            uuid_dir = os.path.join(self.tmp_dir, "buildid", uuid)
            os.makedirs(uuid_dir)
            if debuginfo:
                shutil.copy(self.getBuildArtifact(debuginfo), os.path.join(uuid_dir, "debuginfo"))
            if executable:
                shutil.copy(self.getBuildArtifact(executable), os.path.join(uuid_dir, "executable"))
        os.remove(self.getBuildArtifact("main.dwo"))
        # Configure LLDB for the test:
        self.runCmd("settings set symbols.enable-external-lookup %s" % str(use_debuginfod).lower())
        self.runCmd("settings clear plugin.symbol-locator.debuginfod.server-urls")
        if use_debuginfod:
            self.runCmd("settings set plugin.symbol-locator.debuginfod.cache-path %s/cache" % self.tmp_dir)
            self.runCmd("settings insert-before plugin.symbol-locator.debuginfod.server-urls 0 file://%s" % self.tmp_dir)
