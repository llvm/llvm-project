"""
Test LLDB's handling of qSymbol sequences.
"""

from textwrap import dedent
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase
from lldbsuite.support.seven import hexlify, unhexlify


class MyResponder(MockGDBServerResponder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wanted_symbols = [
            "main",
            "local_address",
            "global_value",
            "local_value",
            "not_a_symbol",
        ]
        self.last_symbol_request = None
        self.symbol_results = {}

    def qSymbol(self, args):
        # args will be <name hex encoded>:<hex value>.
        # In the initial packet both fields are empty.
        if args == ":":
            assert self.last_symbol_request is None
        else:
            # Anything else should be a response to our previous request.
            value, name = args.split(":")
            name = unhexlify(name)
            assert name == self.last_symbol_request

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
        self.server.responder = MyResponder()

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
                # FIXME: Should return a value.
                ("global_value", None),
                # FIXME: Should return a value.
                ("local_value", None),
                ("not_a_symbol", None),
            ]
        )
        self.assertEqual(expected_results, self.server.responder.symbol_results)
