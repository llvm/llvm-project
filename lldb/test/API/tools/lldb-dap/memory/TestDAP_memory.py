"""
Test lldb-dap memory support
"""

from base64 import b64decode

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import *


class TestDAP_memory(DAPTestCaseBase):
    def stop_at_breakpoint(self, session: DAPTestSession):
        """Build, launch, and stop at the `// Breakpoint` line. Returns the
        top frame at that stop."""
        program = self.getBuildArtifact("a.out")
        source = self.getSourcePath("main.cpp")
        bp_line = line_number(source, "// Breakpoint")
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, [bp_line])
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)
        thread_ctx = session.thread_context_from(stop_event)
        return thread_ctx.top_frame()

    @skipIfWindows
    def test_memory_refs_variables(self):
        """Tests memory references on local variables."""
        session = self.build_and_create_session()
        top_frame = self.stop_at_breakpoint(session)
        locals = top_frame.locals

        # Pointers should have memory references.
        self.assertIsNotNone(locals["rawptr"].memoryReference)
        # Non-pointers should also have memory references.
        self.assertIsNotNone(locals["not_a_ptr"].memoryReference)

    @skipIfWindows
    def test_memory_refs_evaluate(self):
        """Tests memory references on `evaluate` responses."""
        session = self.build_and_create_session()
        top_frame = self.stop_at_breakpoint(session)

        eval_body = session.evaluate("rawptr", frameId=top_frame.id)
        self.assertIsNotNone(eval_body.memoryReference)

    @skipIfWindows
    def test_memory_refs_set_variable(self):
        """Tests memory references on `setVariable` responses."""
        session = self.build_and_create_session()
        top_frame = self.stop_at_breakpoint(session)
        locals = top_frame.locals

        ptr_value = locals["rawptr"].variable.value_as_int
        response = locals.set("rawptr", ptr_value + 2)
        response = self.expect_success(response)
        self.assertIsNotNone(response.body.memoryReference)

    @skipIfWindows
    @requireExpressionEvaluation
    def test_readMemory(self):
        """Tests the `readMemory` request."""
        session = self.build_and_create_session()
        top_frame = self.stop_at_breakpoint(session)

        eval_body = session.evaluate("*rawptr", frameId=top_frame.id)
        memref = self.expect_not_none(eval_body.memoryReference)

        # We can read the complete string.
        response = session.read_memory(memref, count=5, offset=0).result()
        data = self.expect_not_none(response.body.data)
        self.assertEqual(b64decode(data), b"dead\0")

        # Large reads return partial results.
        response = session.read_memory(memref, count=4096, offset=0).result()
        data = self.expect_not_none(response.body.data)
        self.assertEqual(b64decode(data)[0:5], b"dead\0")

        # Offsets work.
        response = session.read_memory(memref, count=3, offset=2).result()
        data = self.expect_not_none(response.body.data)
        self.assertEqual(b64decode(data), b"ad\0")

        # Reads of size 0 are successful.
        # VSCode uses these to probe whether a memoryReference can actually be dereferenced.
        response = session.read_memory(memref, count=0, offset=0).result()
        self.assertIsNone(
            response.body.data, f"expects no data in response: {response!r}"
        )

        # Reads at offset 0x0 return unreadable bytes.
        bytes_to_read = 6
        response = session.read_memory("0x0", count=bytes_to_read, offset=0).result()
        self.assertEqual(response.body.unreadableBytes, bytes_to_read)

        # Reads with an invalid address fail.
        session.read_memory("-3204", count=10, offset=0).error(
            "expect fail on reading memory."
        )

        session.continue_to_exit()

    # Flakey on 32-bit Arm Linux.
    @skipIf(oslist=["linux"], archs=["arm$"])
    @requireExpressionEvaluation
    def test_writeMemory(self):
        """Tests the `writeMemory` request."""
        session = self.build_and_create_session()
        top_frame = self.stop_at_breakpoint(session)

        # Get `not_a_ptr`'s writable variable's memory reference.
        eval_body = session.evaluate("not_a_ptr", frameId=top_frame.id)
        memref = self.expect_not_none(eval_body.memoryReference)

        # Write the decimal value 50 (0x32 in hexadecimal) to memory.
        # This corresponds to the ASCII character '2' and encodes to base64
        # as "Mg==".
        response = session.write_memory(memref, value=50, offset=0, allowPartial=True)
        response = self.expect_success(response)
        self.assertEqual(response.body.bytesWritten, 1)

        # Read back and verify.
        read_response = session.read_memory(memref, count=1, offset=0).result()
        self.assertEqual(read_response.body.data, "Mg==")

        # Write the decimal value 100 (0x64 in hexadecimal) to memory with
        # allowPartial=False. This corresponds to the ASCII character 'd' and
        # encodes to base64 as "ZA==".
        response = session.write_memory(memref, value=100, offset=0, allowPartial=False)
        response = self.expect_success(response)
        self.assertEqual(response.body.bytesWritten, 1)

        # Read back and verify.
        read_response = session.read_memory(memref, count=1, offset=0).result()
        self.assertEqual(read_response.body.data, "ZA==")

        # Writing to 0x0 fails.
        response = session.write_memory("0x0", value=50, offset=0, allowPartial=True)
        self.expect_error(response)

        # Writing to a malformed memory reference fails.
        response = session.write_memory("12345", value=50, offset=0, allowPartial=True)
        self.expect_error(response)

        # Writing to a non-writable region returns a not-writable error.
        eval_body = session.evaluate("nonWritable", frameId=top_frame.id)
        nonwritable_ref = self.expect_not_none(eval_body.memoryReference)
        response = session.write_memory(
            nonwritable_ref, value=50, offset=0, allowPartial=False
        )
        err = self.expect_error(response)
        err_msg = self.expect_not_none(err.body and err.body.error)
        self.assertRegex(
            err_msg.format,
            rf"Memory {nonwritable_ref} region is not writable",
        )

        # Writing an empty value (no data) fails.
        response = session.write_memory(nonwritable_ref, value="")
        err = self.expect_error(response)
        err_msg = self.expect_not_none(err.body and err.body.error)
        self.assertRegex(
            err_msg.format,
            r"Data cannot be empty value. Provide valid data",
        )

        # Large writes spanning non-writable regions fail.
        data = bytes([0xFF] * 8192)

        response = session.write_memory(
            nonwritable_ref,
            value=data,
            offset=0,
            allowPartial=False,
        )
        err = self.expect_error(response)
        err_msg = self.expect_not_none(err.body and err.body.error)
        self.assertRegex(
            err_msg.format, rf"Memory {nonwritable_ref} region is not writable"
        )

        session.continue_to_exit()
