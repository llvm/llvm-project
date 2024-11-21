import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


"""
Test support for the DebugInfoD network symbol acquisition protocol.
This one is for simple / no split-dwarf scenarios.

For no-split-dwarf scenarios, there are 2 variations:
1 - A stripped binary with it's corresponding unstripped binary:
2 - A stripped binary with a corresponding --only-keep-debug symbols file
"""


class DebugInfodTests(TestBase):
    # No need to try every flavor of debug inf.
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_normal_no_symbols(self):
        """
        Validate behavior with no symbols or symbol locator.
        ('baseline negative' behavior)
        """
        test_root = self.config_test(["a.out"])
        self.try_breakpoint(False)

    @skipUnlessPlatform(["linux", "freebsd"])
    def test_normal_default(self):
        """
        Validate behavior with symbols, but no symbol locator.
        ('baseline positive' behavior)
        """
        test_root = self.config_test(["a.out", "a.out.debug"])
        self.try_breakpoint(True)

    @skipIfCurlSupportMissing
    @skipUnlessPlatform(["linux", "freebsd"])
    def test_debuginfod_symbols(self):
        """
        Test behavior with the full binary available from Debuginfod as
        'debuginfo' from the plug-in.
        """
        test_root = self.config_test(["a.out"], "a.out.unstripped")
        self.try_breakpoint(True)

    @skipIfCurlSupportMissing
    @skipUnlessPlatform(["linux", "freebsd"])
    def test_debuginfod_executable(self):
        """
        Test behavior with the full binary available from Debuginfod as
        'executable' from the plug-in.
        """
        test_root = self.config_test(["a.out"], None, "a.out.unstripped")
        self.try_breakpoint(True)

    @skipIfCurlSupportMissing
    @skipUnlessPlatform(["linux", "freebsd"])
    def test_debuginfod_okd_symbols(self):
        """
        Test behavior with the 'only-keep-debug' symbols available from Debuginfod.
        """
        test_root = self.config_test(["a.out"], "a.out.debug")
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
        self.assertEqual(
            should_have_loc,
            line_entry != None and line_entry.IsValid(),
            "Loc line entry is valid",
        )
        if should_have_loc:
            self.assertEqual(line_entry.GetLine(), 4)
            self.assertEqual(
                line_entry.GetFileSpec().GetFilename(),
                self.main_source_file.GetFilename(),
            )
        self.dbg.DeleteTarget(target)
        shutil.rmtree(self.tmp_dir)

    def config_test(self, local_files, debuginfo=None, executable=None):
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

        uuid = self.getUUID("a.out")
        if not uuid:
            self.fail("Could not get UUID for a.out")
            return
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.tmp_dir = tempfile.mkdtemp()
        test_dir = os.path.join(self.tmp_dir, "test")
        os.makedirs(test_dir)

        self.aout = ""
        # Copy the files used by the test:
        for f in local_files:
            shutil.copy(self.getBuildArtifact(f), test_dir)
            # The first item is the binary to be used for the test
            if self.aout == "":
                self.aout = os.path.join(test_dir, f)

        use_debuginfod = debuginfo != None or executable != None

        # Populated the 'file://... mocked' Debuginfod server:
        if use_debuginfod:
            os.makedirs(os.path.join(self.tmp_dir, "cache"))
            uuid_dir = os.path.join(self.tmp_dir, "buildid", uuid)
            os.makedirs(uuid_dir)
            if debuginfo:
                shutil.copy(
                    self.getBuildArtifact(debuginfo),
                    os.path.join(uuid_dir, "debuginfo"),
                )
            if executable:
                shutil.copy(
                    self.getBuildArtifact(executable),
                    os.path.join(uuid_dir, "executable"),
                )

        # Configure LLDB for the test:
        self.runCmd(
            "settings set symbols.enable-external-lookup %s"
            % str(use_debuginfod).lower()
        )
        self.runCmd("settings clear plugin.symbol-locator.debuginfod.server-urls")
        if use_debuginfod:
            self.runCmd(
                "settings set plugin.symbol-locator.debuginfod.cache-path %s/cache"
                % self.tmp_dir
            )
            self.runCmd(
                "settings insert-before plugin.symbol-locator.debuginfod.server-urls 0 file://%s"
                % self.tmp_dir
            )

    def getUUID(self, filename):
        try:
            spec = lldb.SBModuleSpec()
            spec.SetFileSpec(lldb.SBFileSpec(self.getBuildArtifact(filename)))
            module = lldb.SBModule(spec)
            uuid = module.GetUUIDString().replace("-", "").lower()
            # Don't want lldb's fake 32 bit CRC's for this one
            return uuid if len(uuid) > 8 else None
        except:
            return None
