"""
Test LLDB's handling of qSymbol sequences.
"""

from pathlib import Path
from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.support.seven import hexlify, unhexlify


class MyResponder(MockGDBServerResponder):
    def __init__(self, test_obj, wanted_symbols=None):
        super().__init__()
        self.wanted_symbols = (
            wanted_symbols
            if wanted_symbols is not None
            else [
                "main",
                "local_address",
                "global_value",
                "local_value",
                "not_a_symbol",
            ]
        )
        self.last_symbol_request = None
        self.symbol_results = {}
        self.test_obj = test_obj

    def qSymbol(self, args):
        # args will be <name hex encoded>:<hex value>.
        # In the initial packet both fields are empty.
        if args == ":":
            self.test_obj.assertIsNone(self.last_symbol_request)
        else:
            # Anything else should be a response to our previous request.
            value, name = args.split(":")
            name = unhexlify(name)
            self.test_obj.assertEqual(name, self.last_symbol_request)

            if value:
                self.symbol_results[name] = int(value, 16)

        if self.wanted_symbols:
            want = self.wanted_symbols.pop(0)
            self.last_symbol_request = want
            self.symbol_results[want] = None

            return "qSymbol:" + hexlify(want)

        # "OK" ends the qSymbol sequence.
        return "OK"


class TestQSymbol(GDBRemoteTestBase):
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_qsymbol(self):
        target = self.createTarget("test_qsymbol.yaml")
        self.server.responder = MyResponder(self)

        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        # LLDB will send a qSymbol shortly after connecting, which starts the sequence.
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # LLDB should have responded in some way to all the qSymbol requests.
        self.assertFalse(self.server.responder.wanted_symbols)

        expected_results = dict(
            [
                ("main", 0x1000),
                ("local_address", 0x1004),
                ("global_value", 0x1234),
                ("local_value", 0xABCD),
                ("not_a_symbol", None),
            ]
        )
        self.assertEqual(expected_results, self.server.responder.symbol_results)

    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_dispatch_symbol_uses_linkage_name(self):
        self.check_dispatch_symbol(has_dispatch_data=True)

    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_missing_dispatch_symbol_rejects_cpp_basename(self):
        self.check_dispatch_symbol(has_dispatch_data=False)

    def check_dispatch_symbol(self, has_dispatch_data):
        self.runCmd("settings set target.preload-symbols false")
        fixture = "test_qsymbol.yaml"
        if not has_dispatch_data:
            missing = Path(self.getBuildArtifact("qsymbol-missing.yaml"))
            missing.write_text(
                (Path(self.getSourceDir()) / fixture)
                .read_text()
                .replace(
                    "Name: dispatch_queue_offsets", "Name: unrelated_dispatch_data"
                )
            )
            fixture = str(missing)
        target = self.createTarget(fixture)
        self.server.responder = MyResponder(
            self, ["dispatch_queue_offsets", "other_wire_request"]
        )
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )
        self.assertFalse(self.server.responder.wanted_symbols)
        # The known dispatch ABI must not resolve to the C++ function basename
        # at 0x1008. Other requests keep their C++ basename lookup behavior.
        self.assertEqual(
            self.server.responder.symbol_results,
            {
                "dispatch_queue_offsets": 0x2000 if has_dispatch_data else None,
                "other_wire_request": 0x1009,
            },
        )
