"""
Tests the jMultiBreakpoint packet, this test runs against whichever debug server
the platform provides (debugserver on macOS, lldb-server elsewhere).
"""

import json

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.gdbclientutils import *


@skipIfWindows  # No server on Windows.
@skipIfOutOfTreeDebugserver
# Runs on systems where we can always predict the software break size
@skipIf(archs=no_match(["x86_64", "arm64", "aarch64"]))
class TestMultiBreakpoint(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_invalid_packet(self, packet_str):
        reply = lldbutil.send_packet_get_reply(self, packet_str)
        if reply.startswith("E"):
            return
        else:
            self.assertMultiResponse(reply, ["error"])

    def send_packet(self, packet_str):
        return lldbutil.send_packet_get_reply(self, packet_str)

    def assertMultiResponse(self, reply, expected):
        """Assert a JSON-array multi-response matches the expected pattern.

        Each element of `expected` is either 'OK' for an exact match, or
        'error' to accept any error response (starting with 'E')."""
        parts = json.loads(reply)["results"]
        self.assertEqual(
            len(parts),
            len(expected),
            f"Expected {len(expected)} responses, got {len(parts)}: {reply}",
        )
        for i, (actual, exp) in enumerate(zip(parts, expected)):
            if exp == "OK":
                self.assertEqual(
                    actual, "OK", f"Response {i}: expected OK, got {actual}"
                )
            elif exp == "error":
                self.assertTrue(
                    actual.startswith("E"),
                    f"Response {i}: expected error, got {actual}",
                )
            else:
                self.fail(f'Bad expected value "{exp}" at index {i}')

    def get_function_address(self, name):
        """Return the hex address of a function as a string (no 0x prefix)."""
        funcs = self.target.FindFunctions(name)
        self.assertGreater(len(funcs), 0, f'Could not find function "{name}"')
        addr = funcs[0].GetSymbol().GetStartAddress().GetLoadAddress(self.target)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)
        return f"{addr:x}"

    def test_multi_breakpoint(self):
        # Debugserver uses refcounted breakpoints
        breakpoints_are_refcounted = self.platformIsDarwin()

        self.build()
        source_file = lldb.SBFileSpec("main.c")
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", source_file
        )

        # Verify the server advertises jMultiBreakpoint support.
        capabilities = lldbutil.get_qsupported_capabilities(self)
        self.assertIn("jMultiBreakpoint+", capabilities)

        addr_a = self.get_function_address("func_a")
        addr_b = self.get_function_address("func_b")
        addr_c = self.get_function_address("func_c")

        # For breakpoint kind, use 4 on AArch64 (4-byte instruction), 1 elsewhere.
        arch = self.getArchitecture()
        if arch in ["arm64", "aarch64"]:
            bp_kind = "4"
        else:
            bp_kind = "1"

        # --- Malformed packets ---
        # Very light error testing, as debugserver and lldb-server behave
        # somewhat differently under malformed input.

        # Empty body after colon.
        self.check_invalid_packet("jMultiBreakpoint:")
        # Not a dictionary
        self.check_invalid_packet((f'jMultiBreakpoint:["Z0,{addr_a},{bp_kind}"]'))

        def make_packet(array):
            key = "breakpoint_requests"
            json_str = json.dumps({key: array})
            return f"jMultiBreakpoint: {json_str}"

        # --- Set a single breakpoint ---
        array = [f"Z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK"])

        # --- Remove the breakpoint we just set ---
        array = [f"z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK"])

        # --- Set multiple breakpoints at once ---
        array = [
            f"Z0,{addr_a},{bp_kind}",
            f"Z0,{addr_b},{bp_kind}",
            f"Z0,{addr_c},{bp_kind}",
        ]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK", "OK"])

        # --- Remove multiple breakpoints at once ---
        array = [
            f"z0,{addr_a},{bp_kind}",
            f"z0,{addr_b},{bp_kind}",
            f"z0,{addr_c},{bp_kind}",
        ]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK", "OK"])

        # --- Mixed set and remove in one packet ---
        # Set two breakpoints first.
        array = [f"Z0,{addr_a},{bp_kind}", f"Z0,{addr_b},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK"])

        # Remove one, set another, remove the other.
        array = [
            f"z0,{addr_a},{bp_kind}",
            f"Z0,{addr_c},{bp_kind}",
            f"z0,{addr_b},{bp_kind}",
        ]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK", "OK"])

        # Clean up.
        array = [f"z0,{addr_c},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK"])

        # --- Set the same breakpoint twice
        array = [f"Z0,{addr_a},{bp_kind}", f"Z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK"])

        # Clean up both.
        array = [f"z0,{addr_a},{bp_kind}", f"z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(
            reply, ["OK", "OK" if breakpoints_are_refcounted else "error"]
        )

        # --- Set the same breakpoint twice, but remove it thrice.
        array = [f"Z0,{addr_a},{bp_kind}", f"Z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK"])
        array = [
            f"z0,{addr_a},{bp_kind}",
            f"z0,{addr_a},{bp_kind}",
            f"z0,{addr_a},{bp_kind}",
        ]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(
            reply, ["OK", "OK" if breakpoints_are_refcounted else "error", "error"]
        )

        # --- Set and remove the same address in a single packet ---
        # The spec requires requests to be executed in order, so the set
        # should succeed and the subsequent remove should find and clear it.
        array = [f"Z0,{addr_a},{bp_kind}", f"z0,{addr_a},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "OK"])

        # --- Remove a breakpoint that was never set ---
        array = [f"z0,{addr_b},{bp_kind}"]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["error"])

        # --- A failure in the middle should not prevent later requests from succeeding ---
        array = [
            f"Z0,{addr_a},{bp_kind}",
            f"z0,{addr_b},{bp_kind}",
            f"z0,{addr_a},{bp_kind}",
        ]
        reply = self.send_packet(make_packet(array))
        self.assertMultiResponse(reply, ["OK", "error", "OK"])
