import http.server
import os
import shutil
import socketserver
import threading
from functools import partial

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
    Context Manager to populate a file structure equivalent to SymStore.exe
    """

    def __init__(self, test, exe, pdb):
        self._test = test
        self._exe = exe
        self._pdb = pdb

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
        symstore_dir = self._test.getBuildArtifact("symstore")
        pdb_dir = os.path.join(symstore_dir, self._pdb, key)
        os.makedirs(pdb_dir, exist_ok=True)
        shutil.move(
            self._test.getBuildArtifact(self._pdb),
            os.path.join(pdb_dir, self._pdb),
        )
        return symstore_dir

    def __exit__(self, *exc_info):
        """
        Reset settings
        """
        self._test.runCmd("settings clear plugin.symbol-locator.symstore")


class HTTPServer:
    """
    Context Manager to serve a local directory tree via HTTP.
    """

    def __init__(self, dir):
        address = ("localhost", 0)  # auto-select free port
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=dir)
        self._server = socketserver.ThreadingTCPServer(address, handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, *exc_info):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join()


class SymStoreTests(TestBase):
    SHARED_BUILD_TESTCASE = False
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
        with MockedSymStore(self, exe, sym) as dir:
            self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {dir}")
            self.try_breakpoint(exe, ext_lookup=False, should_have_loc=False)

    def test_local_dir(self):
        """
        Check that breakpoint resolves with local SymStore.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as dir:
            self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {dir}")
            self.try_breakpoint(exe, should_have_loc=True)

    # TODO: Add test coverage for common HTTPS security scenarios, e.g. self-signed
    # certs, non-HTTPS redirects, etc.
    def test_http(self):
        """
        Check that breakpoint hits with remote SymStore.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as dir:
            with HTTPServer(dir) as url:
                self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {url}")
                self.try_breakpoint(exe, should_have_loc=True)
