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
    def test_read_memory(self):
        """
        Tests the 'read_memory' request
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
        rawptr_ref = locals["rawptr"]["memoryReference"]

        # We can read the complete string
        mem = self.dap_server.request_read_memory(rawptr_ref, 0, 5)["body"]
        self.assertEqual(mem["unreadableBytes"], 0)
        self.assertEqual(b64decode(mem["data"]), b"dead\0")

        # Use an offset
        mem = self.dap_server.request_read_memory(rawptr_ref, 2, 3)["body"]
        self.assertEqual(b64decode(mem["data"]), b"ad\0")

        # Use a negative offset
        mem = self.dap_server.request_read_memory(rawptr_ref, -1, 6)["body"]
        self.assertEqual(b64decode(mem["data"])[1:], b"dead\0")

        # Reads of size 0 are successful
        # VS-Code sends those in order to check if a `memoryReference` can actually be dereferenced.
        mem = self.dap_server.request_read_memory(rawptr_ref, 0, 0)
        self.assertEqual(mem["success"], True)
        self.assertEqual(mem["body"]["data"], "")

        # Reads at offset 0x0 fail
        mem = self.dap_server.request_read_memory("0x0", 0, 6)
        self.assertEqual(mem["success"], False)
        self.assertEqual(mem["message"], "Memory region is not readable")

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
        # Smart pointers also have memory-references
        self.assertIn(
            "memoryReference",
            self.dap_server.get_local_variable_child("smartptr", "pointer").keys(),
        )
        # Non-pointers should not have memory-references
        self.assertNotIn("memoryReference", locals["not_a_ptr"].keys())

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

        # Pointers contain memory references
        self.assertIn(
            "memoryReference",
            self.dap_server.request_evaluate("rawptr + 1")["body"].keys(),
        )

        # Non-pointer expressions don't include a memory reference
        self.assertNotIn(
            "memoryReference",
            self.dap_server.request_evaluate("1 + 3")["body"].keys(),
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

        # Pointers contain memory references
        ptr_value = self.get_local_as_int("rawptr")
        self.assertIn(
            "memoryReference",
            self.dap_server.request_setVariable(1, "rawptr", ptr_value + 2)[
                "body"
            ].keys(),
        )

        # Non-pointer expressions don't include a memory reference
        self.assertNotIn(
            "memoryReference",
            self.dap_server.request_setVariable(1, "not_a_ptr", 42)["body"].keys(),
        )
