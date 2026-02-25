import glob
import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

"""
Test debug symbol acquisition from a local SymStore repository. We populate the
respective file structure in a temporary directory and run LLDB on it. This is
supposed to work cross-platform. The test can run on all platforms that can link
debug info in a PDB file with clang.
"""

class SymStoreLocalPDBTests(TestBase):
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

    def build_inferior_with_pdb(self):
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.build()
        pdbs = glob.glob(os.path.join(self.getBuildDir(), "*.pdb"))
        return len(pdbs) > 0

    def populate_symstore(self, tmp):
        """
        Mock local symstore directory tree and fill in build artifacts:
        * tmp/test/<exe>
        * tmp/server/<pdb>/<key>/<pdb>
        """
        binary_name = "a.out"
        pdb_name = "a.pdb"
        key = self.symstore_key(binary_name)
        if key is None:
            self.skipTest("Binary has no valid UUID for PDB")

        # Move exe to isolated directory
        test_dir = os.path.join(tmp, "test")
        os.makedirs(test_dir)
        shutil.move(self.getBuildArtifact(binary_name), test_dir)
        self.aout = os.path.join(test_dir, binary_name)

        # Move PDB to SymStore directory
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

    # TODO: Add a test that fails if we don't set the URL
    def test_basic(self):
        """Check that breakpoint hits if LLDB fetches PDB from local SymStore"""
        if not self.build_inferior_with_pdb():
            self.skipTest("Build did not produce a PDB file")
            
        tmp_dir = tempfile.mkdtemp()
        symstore_dir = self.populate_symstore(tmp_dir)

        self.runCmd(
            "settings set plugin.symbol-locator.symstore.urls %s"
            % symstore_dir.replace("\\", "/")
        )

        self.try_breakpoint(should_have_loc=True)
        shutil.rmtree(tmp_dir)
