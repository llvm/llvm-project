import glob
import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MicrosoftSymSrvTests(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

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
        Build test binary and mock local symstore directory tree:
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
        key = self.symstore_key(binary_name)
        if key is None:
            self.skipTest("Could not obtain a 20-byte PDB UUID from the binary")

        # Set up test directory with just the binary (no PDB).
        test_dir = os.path.join(tmp, "test")
        os.makedirs(test_dir)
        shutil.move(self.getBuildArtifact(binary_name), test_dir)
        self.aout = os.path.join(test_dir, binary_name)

        # SymStore directory tree: <pdb>/<key>/<pdb>
        server_dir = os.path.join(tmp, "server")
        pdb_key_dir = os.path.join(server_dir, pdb_name, key)
        os.makedirs(pdb_key_dir)
        shutil.move(
            self.getBuildArtifact(pdb_name),
            os.path.join(pdb_key_dir, pdb_name),
        )

        return server_dir

    def symstore_key(self, exe):
        """Load module UUID like: 12345678-1234-5678-9ABC-DEF012345678-00000001
        and transform to SymStore key: 12345678123456789ABCDEF0123456781"""
        try:
            spec = lldb.SBModuleSpec()
            spec.SetFileSpec(lldb.SBFileSpec(self.getBuildArtifact(exe)))
            module = lldb.SBModule(spec)
            raw = module.GetUUIDString().replace("-", "").upper()
            if len(raw) != 40:
                return None
            guid_hex = raw[:32]
            age = int(raw[32:], 16)
            return guid_hex + str(age)
        except Exception:
            return None

    # TODO: Check on other platforms, it should work in theory
    @skipUnlessPlatform(["windows"])
    def test_local_folder(self):
        """Check that LLDB can fetch PDB from local SymStore directory"""
        tmp_dir = tempfile.mkdtemp()
        symstore_dir = self.populate_symstore(tmp_dir)

        self.runCmd(
            "settings set plugin.symbol-locator.microsoft.symstore-urls %s"
            % symstore_dir.replace("\\", "/")
        )

        self.try_breakpoint(should_have_loc=True)
        shutil.rmtree(tmp_dir)
