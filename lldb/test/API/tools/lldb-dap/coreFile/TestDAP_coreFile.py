"""
Test lldb-dap coreFile attaching
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import (
    AttachArgs,
    ContinueArgs,
    NextArgs,
    Source,
    StackFrame,
)

# The expected backtrace when loading the bundled linux-x86_64.core. Shared by
# the tests that load this core through different mechanisms (the "coreFile"
# attach key and "attachCommands") so we can assert they behave identically.
EXPECTED_CORE_FRAMES = [
    StackFrame(
        column=0,
        id=524288,
        line=4,
        moduleId="01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
        name="bar",
        source=Source(
            name="main.c",
            path="/home/labath/test/main.c",
            presentationHint="deemphasize",
        ),
        instructionPointerReference="0x40011C",
    ),
    StackFrame(
        column=0,
        id=524289,
        line=10,
        moduleId="01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
        name="foo",
        source=Source(
            name="main.c",
            path="/home/labath/test/main.c",
            presentationHint="deemphasize",
        ),
        instructionPointerReference="0x400142",
    ),
    StackFrame(
        column=0,
        id=524290,
        line=16,
        moduleId="01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
        name="_start",
        source=Source(
            name="main.c",
            path="/home/labath/test/main.c",
            presentationHint="deemphasize",
        ),
        instructionPointerReference="0x40015F",
    ),
]


class TestDAP_coreFile(DAPTestCaseBase):
    @skipIfLLVMTargetMissing("X86")
    def test_core_file(self):
        exe_file = self.getSourcePath("linux-x86_64.out")
        core_file = self.getSourcePath("linux-x86_64.core")

        session = self.create_session()
        process_event = session.attach(AttachArgs(program=exe_file, coreFile=core_file))
        stop_event = session.wait_for_stopped_event(after=process_event)
        thread_id = self.expect_not_none(stop_event.body.threadId)

        frames = session.stack_trace(thread_id).body.stackFrames
        self.assertEqual(frames, EXPECTED_CORE_FRAMES)

        # Resuming a core process should fail. the process stays stopped
        # with the same backtrace.
        session.send_request(ContinueArgs(thread_id)).error()
        frames = session.stack_trace(thread_id).body.stackFrames
        self.assertEqual(frames, EXPECTED_CORE_FRAMES)

        # Same for step-over.
        session.send_request(NextArgs(threadId=thread_id)).error()
        frames = session.stack_trace(thread_id).body.stackFrames
        self.assertEqual(frames, EXPECTED_CORE_FRAMES)

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_attach_commands(self):
        """Loading a core through "attachCommands" (e.g. `target create --core`)
        should behave identically to using the "coreFile" attach key: the
        session stops with the real crash reason and cannot be resumed."""
        exe_file = self.getSourcePath("linux-x86_64.out")
        core_file = self.getSourcePath("linux-x86_64.core")

        session = self.create_session()
        # Bootstrap the core target purely through a custom attach command,
        # mirroring how the "coreFile" key passes the same program.
        # configurationDone must succeed: a core is a non-live session, so the
        # adapter must not try to resume it (resuming a core fails).
        process_event = session.attach(
            AttachArgs(
                program=exe_file,
                attachCommands=[f"target create --core '{core_file}' '{exe_file}'"],
            )
        )

        # The stop must be reported with the real crash reason, not "entry".
        stop_event = session.verify_stopped_on_exception(after=process_event)
        thread_id = self.expect_not_none(stop_event.body.threadId)

        # The backtrace must match the "coreFile" attach key exactly.
        frames = session.stack_trace(thread_id).body.stackFrames
        self.assertEqual(frames, EXPECTED_CORE_FRAMES)

        # Resuming should fail.
        session.send_request(ContinueArgs(thread_id)).error()
        frames = session.stack_trace(thread_id).body.stackFrames
        self.assertEqual(frames, EXPECTED_CORE_FRAMES)

    def test_wrong_core_file(self):
        """Attaching with a file that isn't a real core should fail cleanly
        during configurationDone rather than crashing the adapter."""
        exe_file = self.getSourcePath("linux-x86_64.out")
        wrong_core_file = self.getSourcePath("main.c")

        session = self.create_session()
        session.initialize_sequence(session.initialize_args)
        pending_attach = session.send_request(
            AttachArgs(program=exe_file, coreFile=wrong_core_file)
        )
        session.verify_configuration_done(expected_success=False)

        resp = pending_attach.error()
        resp_error = self.expect_not_none(resp.body and resp.body.error)
        self.assertEqual(resp_error.format, "Failed to create the process")

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_stopped_reason(self):
        """The stopped event for a core file should report the actual crash
        reason (e.g. 'exception') rather than 'entry'."""
        exe_file = self.getSourcePath("linux-x86_64.out")
        core_file = self.getSourcePath("linux-x86_64.core")

        session = self.create_session()
        process_event = session.attach(AttachArgs(program=exe_file, coreFile=core_file))

        stop_event = session.verify_stopped_on_exception(after=process_event)
        self.assertIsNotNone(stop_event.body.description, "expect a stop description.")

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_array(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        exe_file = self.getSourcePath("linux-x86_64.out")
        core_file = self.getSourcePath("linux-x86_64.core")
        current_dir = self.getSourceDir()

        session = self.create_session()
        process_event = session.attach(
            AttachArgs(
                program=exe_file,
                coreFile=core_file,
                sourceMap=[("/home/labath/test", current_dir)],
            )
        )

        stop_event = session.verify_stopped_on_exception(after=process_event)
        top_frame = session.top_frame_from(stop_event).frame
        top_source = self.expect_not_none(top_frame.source and top_frame.source.path)
        self.assertIn(current_dir, top_source)

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_object(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        exe_file = self.getSourcePath("linux-x86_64.out")
        core_file = self.getSourcePath("linux-x86_64.core")
        current_dir = self.getSourceDir()

        session = self.create_session()
        process_event = session.attach(
            AttachArgs(
                program=exe_file,
                coreFile=core_file,
                sourceMap={"/home/labath/test": current_dir},
            )
        )

        stop_event = session.verify_stopped_on_exception(after=process_event)
        top_frame = session.top_frame_from(stop_event).frame
        top_source = self.expect_not_none(top_frame.source and top_frame.source.path)
        self.assertIn(current_dir, top_source)
