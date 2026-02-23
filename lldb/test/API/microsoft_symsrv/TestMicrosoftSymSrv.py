import glob
import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


"""
Test support for the Microsoft symbol server (symsrv) protocol.

LLDB's SymbolLocatorMicrosoft plugin locates PDB files from symbol servers
that follow the Microsoft symsrv directory layout:

  <store>/<pdb-name>/<GUID-uppercase-no-dashes><age-decimal>/<pdb-name>

The symstore-urls setting accepts entries in SRV*<cache>*<server> notation,
matching the _NT_SYMBOL_PATH convention used by WinDbg and other Microsoft
debuggers.
"""

import debugpy
debugpy.listen(("127.0.0.1", 5678))
debugpy.wait_for_client()
debugpy.breakpoint()

class MicrosoftSymSrvTests(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessPlatform(["windows"])
    def test_local_folder(self):
        """Verify that LLDB fetches the PDB from a local SymStore directory."""
        tmp_dir = tempfile.mkdtemp()
        symstore_dir = self.populate_symstore(tmp_dir)
        
        self.runCmd(
            "settings set plugin.symbol-locator.microsoft.symstore-urls %s" %
            symstore_dir.replace("\\", "/")
        )

        self.try_breakpoint(should_have_loc=True)
        #shutil.rmtree(tmp_dir)
        print(tmp_dir)

    def try_breakpoint(self, should_have_loc):
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
            line_entry is not None and line_entry.IsValid(),
            "Loc line entry validity",
        )
        if should_have_loc:
            self.assertEqual(line_entry.GetLine(), 2)
            self.assertEqual(
                line_entry.GetFileSpec().GetFilename(),
                self.main_source_file.GetFilename(),
            )
        self.dbg.DeleteTarget(target)

    def populate_symstore(self, tmp):
        """
        Build test binary, mock local symstore directory tree:
          tmp/test/a.out                 binary (no PDB in search path)
          tmp/server/<pdb>/<key>/<pdb>   PDB in symstore
        """
        self.build()
        pdbs = glob.glob(os.path.join(self.getBuildDir(), "*.pdb"))
        if len(pdbs) == 0:
            self.skipTest("Build did not produce a PDB file")

        self.main_source_file = lldb.SBFileSpec("main.c")

        binary_name = "a.out"
        pdb_name = "a.pdb"
        uuid_str = self._get_uuid(binary_name)
        if uuid_str is None:
            self.skipTest("Could not obtain a 20-byte PDB UUID from the binary")

        symsrv_key = self._uuid_to_symsrv_key(uuid_str)

        # Set up test directory with just the binary (no PDB).
        test_dir = os.path.join(tmp, "test")
        os.makedirs(test_dir)
        shutil.move(self.getBuildArtifact(binary_name), test_dir)
        self.aout = os.path.join(test_dir, binary_name)

        # SymStore directory tree: <pdb>/<key>/<pdb>
        server_dir = os.path.join(tmp, "server")
        pdb_key_dir = os.path.join(server_dir, pdb_name, symsrv_key)
        os.makedirs(pdb_key_dir)
        shutil.move(
            self.getBuildArtifact(pdb_name),
            os.path.join(pdb_key_dir, pdb_name),
        )

        return server_dir

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    #def _get_pdb_name(self):
    #    """Return the basename of the first PDB in the build directory."""
    #    pdbs = glob.glob(os.path.join(self.getBuildDir(), "*.pdb"))
    #    return os.path.basename(pdbs[0]) if pdbs else None

    def _get_uuid(self, binary_name):
        """Return the UUID string (dashes removed, lowercase) for binary_name,
        or None if it is not a 20-byte PDB UUID."""
        try:
            spec = lldb.SBModuleSpec()
            spec.SetFileSpec(
                lldb.SBFileSpec(self.getBuildArtifact(binary_name))
            )
            module = lldb.SBModule(spec)
            raw = module.GetUUIDString().replace("-", "").lower()
            return raw if len(raw) == 40 else None
        except Exception:
            return None

    @staticmethod
    def _uuid_to_symsrv_key(uuid_lower_40):
        """Convert a 40-char lowercase hex UUID string to a Microsoft symsrv
        key: uppercase GUID (32 chars) followed by decimal age."""
        upper = uuid_lower_40.upper()
        guid_hex = upper[:32]
        age = int(upper[32:], 16)
        return guid_hex + str(age)
