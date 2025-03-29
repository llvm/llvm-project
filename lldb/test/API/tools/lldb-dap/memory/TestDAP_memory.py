"""
Test lldb-dap memory support
"""

from base64 import b64decode
import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_memory(lldbdap_testcase.DAPTestCaseBase):
    def test_memory_refs_variables(self):
        """
        Tests memory references for evaluate
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Breakpoint")],
        )
        self.continue_to_next_stop()

        locals = {l["name"]: l for l in self.dap_server.get_local_variables()}

        # Pointers should have memory-references
        self.assertIn("memoryReference", locals["rawptr"].keys())
        # Non-pointers should also have memory-references
        self.assertIn("memoryReference", locals["not_a_ptr"].keys())

    def test_memory_refs_evaluate(self):
        """
        Tests memory references for evaluate
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Breakpoint")],
        )
        self.continue_to_next_stop()

        self.assertIn(
            "memoryReference",
            self.dap_server.request_evaluate("rawptr")["body"].keys(),
        )

    def test_memory_refs_set_variable(self):
        """
        Tests memory references for `setVariable`
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Breakpoint")],
        )
        self.continue_to_next_stop()

        ptr_value = self.get_local_as_int("rawptr")
        self.assertIn(
            "memoryReference",
            self.dap_server.request_setVariable(1, "rawptr", ptr_value + 2)[
                "body"
            ].keys(),
        )

    def test_readMemory(self):
        """
        Tests the 'readMemory' request
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Breakpoint")],
        )
        self.continue_to_next_stop()

        ptr_deref = self.dap_server.request_evaluate("*rawptr")["body"]
        memref = ptr_deref["memoryReference"]

        # We can read the complete string
        mem = self.dap_server.request_readMemory(memref, 0, 5)["body"]
        self.assertEqual(b64decode(mem["data"]), b"dead\0")

        # We can read large chunks, potentially returning partial results
        mem = self.dap_server.request_readMemory(memref, 0, 4096)["body"]
        self.assertEqual(b64decode(mem["data"])[0:5], b"dead\0")

        # Use an offset
        mem = self.dap_server.request_readMemory(memref, 2, 3)["body"]
        self.assertEqual(b64decode(mem["data"]), b"ad\0")

        # Reads of size 0 are successful
        # VS Code sends those in order to check if a `memoryReference` can actually be dereferenced.
        mem = self.dap_server.request_readMemory(memref, 0, 0)
        self.assertEqual(mem["success"], True)
        self.assertEqual(mem["body"]["data"], "")

        # Reads at offset 0x0 fail
        mem = self.dap_server.request_readMemory("0x0", 0, 6)
        self.assertEqual(mem["success"], False)

    def test_writeMemory(self):
        """
        Tests the 'writeMemory' request
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [line_number(source, "// Breakpoint")],
        )
        self.continue_to_next_stop()

        # Get the 'not_a_ptr' writable variable reference address.
        ptr_deref = self.dap_server.request_evaluate("not_a_ptr")["body"]
        memref = ptr_deref["memoryReference"]

        # Write the Base64-encoded string "Mg==", which decodes to binary 0x32
        # which is decimal 50 and corresponds to the ASCII character '2'.
        mem_response = self.writeMemory(memref, 50, 0, True)
        self.assertEqual(mem_response["success"], True)
        self.assertEqual(mem_response["body"]["bytesWritten"], 1)

        # Read back the modified memory and verify that the written data matches
        # the expected result.
        mem_response = self.dap_server.request_readMemory(memref, 0, 1)
        self.assertEqual(mem_response["success"], True)
        self.assertEqual(mem_response["body"]["data"], "Mg==")

        # Memory write failed for 0x0.
        mem_response = self.writeMemory("0x0", 50, 0, True)
        self.assertEqual(mem_response["success"], False)

        # Malformed memory reference.
        mem_response = self.writeMemory("12345", 50, 0, True)
        self.assertEqual(mem_response["success"], False)

        ptr_deref = self.dap_server.request_evaluate("nonWritable")["body"]
        memref = ptr_deref["memoryReference"]

        # Writing to non-writable region should return an appropriate error.
        mem_response = self.writeMemory(memref, 50, 0, False)
        self.assertEqual(mem_response["success"], False)
        self.assertRegex(
            mem_response["message"],
            r"Memory " + memref + " region is not writable",
        )

        # Trying to write empty value; data=""
        mem_response = self.writeMemory(memref)
        self.assertEqual(mem_response["success"], False)
        self.assertRegex(
            mem_response["message"],
            r"Data cannot be empty value. Provide valid data",
        )
