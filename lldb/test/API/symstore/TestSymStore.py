import http.server
import os
import shutil
import socketserver
import sys
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
        self.cache_dir = test.getBuildArtifact("cache")

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
        # We always configure a valid fallback cache, because we might not have
        # permission to write outside the test directory.
        self._test.runCmd(
            f"settings set plugin.symbol-locator.symstore.cache {self.cache_dir}"
        )
        return symstore_dir

    def __exit__(self, *exc_info):
        """
        Reset settings
        """
        self._test.runCmd("settings clear plugin.symbol-locator.symstore")


class NtSymbolPath:
    """
    Context Manager to temporarily set the _NT_SYMBOL_PATH environment variable.
    """

    def __init__(self, value):
        self._value = value
        self._saved = None

    def __enter__(self):
        self._saved = os.environ.get("_NT_SYMBOL_PATH")
        os.environ["_NT_SYMBOL_PATH"] = self._value

    def __exit__(self, *exc_info):
        if self._saved is None:
            os.environ.pop("_NT_SYMBOL_PATH", None)
        else:
            os.environ["_NT_SYMBOL_PATH"] = self._saved


class HTTPServer:
    """
    Context Manager to serve a local directory tree via HTTP.
    """

    def __init__(self, dir, handler=None):
        address = ("localhost", 0)  # auto-select free port
        if handler is None:
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


class RequestCounter(http.server.SimpleHTTPRequestHandler):
    requests = 0  # class-level so all instances share one counter

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        RequestCounter.requests += 1
        super().do_GET()


class SymStoreTests(TestBase):
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

    def assertFiles(self, dir, expected):
        actual = sum(len(f) for _, _, f in os.walk(dir))
        self.assertEqual(actual, expected)

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

    def test_http_not_found(self):
        """
        Check that we don't issue a warning for a 404 response from a symbol server.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as symstore_dir:
            os.makedirs(f"{symstore_dir}_empty", exist_ok=False)
            with HTTPServer(f"{symstore_dir}_empty") as url:
                self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {url}")
                warnings = ""
                with open(self.getBuildArtifact("stderr.txt"), "w+b") as err_file:
                    self.dbg.SetErrorFileHandle(err_file, False)
                    self.try_breakpoint(exe, should_have_loc=False)
                    self.dbg.SetErrorFileHandle(sys.stderr, False)
                    err_file.seek(0)
                    warnings = err_file.read().decode()
                self.assertEqual(warnings, "")

    # TODO: Add test coverage for common HTTPS security scenarios, e.g. self-signed
    # certs, non-HTTPS redirects, etc.
    def test_http(self):
        """
        Check that breakpoint resolves with remote SymStore.
        """
        exe, sym = self.build_inferior()
        with MockedSymStore(self, exe, sym) as dir:
            with HTTPServer(dir) as url:
                self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {url}")
                self.try_breakpoint(exe, should_have_loc=True)

    def test_sympath_local_dir(self):
        """
        Check that breakpoint resolves with plain directory in _NT_SYMBOL_PATH.
        The PDB is not copied to the configured cache.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with NtSymbolPath(dir):
                self.try_breakpoint(exe, should_have_loc=True)
            self.assertFiles(symstore.cache_dir, 0)

    def test_sympath_local_srv(self):
        """
        Check that breakpoint resolves with local directory in server notation
        in _NT_SYMBOL_PATH. The PDB is not copied to the configured cache.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with NtSymbolPath(f"srv*{dir}"):
                self.try_breakpoint(exe, should_have_loc=True)
            self.assertFiles(symstore.cache_dir, 0)

    def test_sympath_srv(self):
        """
        Check that breakpoint resolves with an HTTP symbol server in _NT_SYMBOL_PATH
        using the srv* notation. The PDB is stored in the configured cache.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            self.assertFiles(symstore.cache_dir, 0)
            with HTTPServer(dir) as url:
                with NtSymbolPath(f"srv*{url}"):
                    self.try_breakpoint(exe, should_have_loc=True)
            key = symstore.get_key_pdb(exe)
            cache_file = os.path.join(symstore.cache_dir, sym, key, sym)
            self.assertTrue(os.path.isfile(cache_file))
            self.assertFiles(symstore.cache_dir, 1)

    def test_sympath_cache_explicit(self):
        """
        Check PDB storage with explicit cache in _NT_SYMBOL_PATH.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with HTTPServer(dir) as url:
                explicit_cache = self.getBuildArtifact("explicit_cache")
                with NtSymbolPath(f"srv*{explicit_cache}*{url}"):
                    self.try_breakpoint(exe, should_have_loc=True)
                self.assertFiles(symstore.cache_dir, 0)
                self.assertFiles(explicit_cache, 1)

    def test_sympath_cache_implicit(self):
        """
        Check PDB storage with implicit cache in _NT_SYMBOL_PATH.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with HTTPServer(dir) as url:
                implicit_cache = self.getBuildArtifact("implicit_cache")
                with NtSymbolPath(f"cache*{implicit_cache};srv*{url}"):
                    self.try_breakpoint(exe, should_have_loc=True)
                self.assertFiles(symstore.cache_dir, 0)
                self.assertFiles(implicit_cache, 1)

    def test_sympath_cache_invalid(self):
        """
        Check that PDB is stored in configured default cache
        if path in _NT_SYMBOL_PATH is invalid.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with HTTPServer(dir) as url:
                invalid_cache = ":\\<invalid_path>"
                self.assertFiles(symstore.cache_dir, 0)
                with NtSymbolPath(f"cache*{invalid_cache};srv*{url}"):
                    self.try_breakpoint(exe, should_have_loc=True)
                self.assertFiles(symstore.cache_dir, 1)

    def test_sympath_cache_empty(self):
        """
        Check that PDB is stored in configured default cache
        if path in _NT_SYMBOL_PATH is empty.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        with symstore as dir:
            with HTTPServer(dir) as url:
                self.assertFiles(symstore.cache_dir, 0)
                with NtSymbolPath(f"cache*;srv*{url}"):
                    self.try_breakpoint(exe, should_have_loc=True)
                self.assertFiles(symstore.cache_dir, 1)

    def test_lookup_order(self):
        """
        Check that _NT_SYMBOL_PATH takes precedence over symstore.urls setting.
        """
        exe, sym = self.build_inferior()
        symstore = MockedSymStore(self, exe, sym)
        RequestCounter.requests = 0
        with symstore as dir:
            with HTTPServer(dir, RequestCounter) as url:
                self.runCmd(f"settings set plugin.symbol-locator.symstore.urls {url}")
                with NtSymbolPath(dir):
                    self.try_breakpoint(exe, should_have_loc=True)
            self.assertEqual(RequestCounter.requests, 0)
