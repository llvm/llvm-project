import os
import shutil
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


"""
Test debug symbol acquisition from a local SymStore repository. This can work
cross-platform and for arbitrary debug info formats. We only support PDB
currently.
"""


class MockedSymStore:
    """
    Context Manager to populate a file structure equivalent to SymStore.exe in a
    temporary directory.
    """

    def __init__(self, test, exe, pdb):
        self._test = test
        self._exe = exe
        self._pdb = pdb
        self._tmp = None

    def get_key_pdb(self, exe):
        """
        Module UUID: 12345678-1234-5678-9ABC-DEF012345678-00000001
        To SymStore key: 12345678123456789ABCDEF0123456781
        """
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(self._test.getBuildArtifact(exe)))
        module = lldb.SBModule(spec)
        raw = module.GetUUIDString().replace("-", "").upper()
        if len(raw) != 40:
            raise RuntimeError("Unexpected number of bytes in embedded UUID")
        guid_hex = raw[:32]
        age = int(raw[32:], 16)
        return guid_hex + str(age)

    def __enter__(self):
        """
        Mock local symstore directory tree, move PDB there and report path.
        """
        key = None
        if self._test.getDebugInfo() == "pdb":
            key = self.get_key_pdb(self._exe)
        self._test.assertIsNotNone(key)
        self._tmp = self._test.getBuildArtifact("tmp")
        pdb_dir = os.path.join(self._tmp, self._pdb, key)
        os.makedirs(pdb_dir)
        shutil.move(
            self._test.getBuildArtifact(self._pdb),
            os.path.join(pdb_dir, self._pdb),
        )
        return self._tmp

    def __exit__(self, *exc_info):
        """
        Clean up and delete original exe so next make won't skip link command.
        """
        shutil.rmtree(self._tmp)
        os.remove(self._test.getBuildArtifact(self._exe))
        self._test.runCmd("settings clear plugin.symbol-locator.symstore")


class SymStoreLocalTests(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def build_inferior(self):
        if self.getDebugInfo() != "pdb":
            self.skipTest("Non-PDB debug info variants not yet supported")
        self.build()
        exe_file = "a.out"
        sym_file = "a.pdb"
        self.assertTrue(os.path.isfile(self.getBuildArtifact(exe_file)))
        self.assertTrue(os.path.isfile(self.getBuildArtifact(sym_file)))
        return exe_file, sym_file

    def try_breakpoint(self, exe, should_have_loc, ext_lookup=True):
        enable = "true" if ext_lookup else "false"
        self.runCmd(f"settings set symbols.enable-external-lookup {enable}")
        target = self.dbg.CreateTarget(self.getBuildArtifact(exe))
        self.assertTrue(target and target.IsValid(), "Target is valid")
        bp = target.BreakpointCreateByName("func")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")
        self.assertEqual(bp.GetNumLocations(), 1 if should_have_loc else 0)
        self.dbg.DeleteTarget(target)

    def test_no_symstore_url(self):
        """
        Check that breakpoint doesn't resolve without SymStore.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym):
            self.try_breakpoint(exe, should_have_loc=False)

    def test_external_lookup_off(self):
        """
        Check that breakpoint doesn't resolve with external lookup disabled.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as symstore_dir:
            self.runCmd(
                f"settings set plugin.symbol-locator.symstore.urls {symstore_dir}"
            )
            self.try_breakpoint(exe, ext_lookup=False, should_have_loc=False)

    def test_local_dir(self):
        """
        Check that breakpoint resolves with local SymStore.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as symstore_dir:
            self.runCmd(
                f"settings set plugin.symbol-locator.symstore.urls {symstore_dir}"
            )
            self.try_breakpoint(exe, should_have_loc=True)
