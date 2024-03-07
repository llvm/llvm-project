"""
Test support for the DebugInfoD network symbol acquisition protocol.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

def getUUID(aoutuuid):
    """
    Pull the 20 byte UUID out of the .note.gnu.build-id section that was dumped
    to a file already, as part of the build.
    """
    import struct
    with open(aoutuuid, "rb") as f:
        data = f.read(36)
        if len(data) != 36:
            return None
        header = struct.unpack_from("<4I", data)
        if len(header) != 4:
            return None
        # 4 element 'prefix', 20 bytes of uuid, 3 byte long string, 'GNU':
        if header[0] != 4 or header[1] != 20 or header[2] != 3 or header[3] != 0x554e47:
            return None
        return data[16:].hex()

def config_test(local_files, uuid, debuginfo, executable):
    """
    Set up a test with local_files[] copied to a particular location
    so that we control which files are, or are not, found in the file system.
    Also, create a stand-alone file-system 'hosted' debuginfod server for the
    given UUID.

    Make the filesystem look like:

    /tmp/<tmpdir>/test/[local_files]

    /tmp/<tmpdir>/cache (for lldb to use as a temp cache)

    /tmp/<tmpdir>/buildid/<uuid>/executable -> <executable>
    /tmp/<tmpdir>/buildid/<uuid>/debuginfo -> <debuginfo>
    Returns the /tmp/<tmpdir> path
    """
    import os
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    test_dir = os.path.join(tmp_dir, "test")
    uuid_dir = os.path.join(tmp_dir, "buildid", uuid)

    # Make the 3 directories
    os.makedirs(os.path.join(tmp_dir, "cache"))
    os.makedirs(uuid_dir)
    os.makedirs(test_dir)

    # Copy the files used by the test:
    for f in local_files:
        shutil.copy(f, test_dir)

    # Fill in the 'file://...' mocked Debuginfod server
    if debuginfo:
        shutil.move(debuginfo, os.path.join(uuiddir, "debuginfo"))
    if executable:
        shutil.move(executable, os.path.join(uuiddir, "executable"))

    return tmp_dir


# Need to test 5 different scenarios:
# 1 - A stripped binary with it's corresponding unstripped binary:
# 2 - A stripped binary with a corresponding --only-keep-debug symbols file
# 3 - A split binary with it's corresponding DWP file
# 4 - A stripped, split binary with an unstripped binary and a DWP file
# 5 - A stripped, split binary with an --only-keep-debug symbols file and a DWP file

class DebugInfodTests(TestBase):
    # No need to try every flavor of debug inf.
    NO_DEBUG_INFO_TESTCASE = True

    def test_stuff(self):
        """This should test stuff."""
        self.build()
        # Pull the UUID out of the binary.
        uuid_file = self.getBuildArtifact("a.out.uuid")
        self.aout = self.getBuildArtifact("a.out")
        self.debugbin = self.getBuildArtifact("a.out.debug")
        self.dwp = self.getBuildArtifact("a.out.dwp")
        self.uuid = getUUID(uuid_file)
        # Setup the fake DebugInfoD server.
        server_root = config_fake_debuginfod_server(self.uuid, self.dwp, self.debugbin)

        # Configure LLDB properly
        self.runCmd("settings set symbols.enable-external-lookup true")
        self.runCmd("settings set plugin.symbol-locator.debuginfod.cache-path %s/cache" % server_root)
        self.runCmd("settings clear plugin.symbol-locator.debuginfod.server-urls")
        cmd = "settings insert-before plugin.symbol-locator.debuginfod.server-urls 0 file://%s" % server_root
        self.runCmd(cmd)
        print(cmd)
        # Check to see if the symbol file is properly loaded
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.sample_test()

    def sample_test(self):
        """You might use the test implementation in several ways, say so here."""
        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        target = target = self.dbg.CreateTarget(self.aout)
        self.assertTrue(target and target.IsValid(), "Target is valid")
        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1)
        print("Loc @ Index 0:")
        print(bp.GetLocationAtIndex(0))
        loc = bp.GetLocationAtIndex(0)
        self.assertTrue(loc and loc.IsValid(), "Location is valid")
        addr = loc.GetAddress()
        self.assertTrue(addr and addr.IsValid(), "Loc address is valid")
        line_entry = addr.GetLineEntry()
        self.assertTrue(line_entry and line_entry.IsValid(), "Loc line entry is valid")
        self.assertEqual(line_entry.GetLine(), 18)
        self.assertEqual(loc.GetLineEntry().GetFileSpec().GetFilename(), self.main_source_file.GetFilename())

    def sample_test_no_launch(self):
        """Same as above but doesn't launch a process."""

        target = self.createTestTarget()
        self.expect_expr("global_test_var", result_value="10")
