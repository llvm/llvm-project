"""
Test lldb-dap logpoints feature.
"""

import os

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import (
    LaunchArgs,
    SourceBreakpoint,
    StoppedEvent,
)


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_logpoints(DAPTestCaseBase):
    def setUp(self):
        DAPTestCaseBase.setUp(self)

        self.main_basename = "main-copy.cpp"
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    def stop_at_before_loop_line(self, session: DAPTestSession) -> StoppedEvent:
        """Launch, set a breakpoint at 'before loop' line and stop there"""
        before_loop_line = line_number("main.cpp", "// before loop")
        program = self.getBuildArtifact("a.out")
        with session.configure(LaunchArgs(program)) as ctx:
            [bp] = session.resolve_source_breakpoints(
                self.main_path, [before_loop_line]
            )

        return session.verify_stopped_on_breakpoint(bp, after=ctx.process_event)

    @skipIfWindows
    def test_logMessage_basic(self):
        """Tests breakpoint logMessage basic functionality."""
        session = self.build_and_create_session()
        initial_stop = self.stop_at_before_loop_line(session)
        source = self.getSourcePath("main.cpp")
        loop_line = line_number(source, "// break loop")
        after_loop_line = line_number(source, "// after loop")

        # Set two breakpoints:
        # 1. First at the loop line with logMessage.
        # 2. Second guard breakpoint at a line after loop.
        logMessage_prefix = "This is log message for { -- "
        logMessage = logMessage_prefix + "{i + 3}, {message}"
        [_, post_loop_breakpoint_id] = session.resolve_source_breakpoints(
            self.main_path,
            [
                SourceBreakpoint(loop_line, logMessage=logMessage),
                SourceBreakpoint(after_loop_line),
            ],
        )

        # Continue and verify we hit the breakpoint after loop line.
        post_loop_stop = session.continue_to_breakpoint(post_loop_breakpoint_id)

        captured = session.collect_console(after=initial_stop, until=post_loop_stop)
        logMessage_output = [
            line
            for line in captured.seen_texts.splitlines()
            if line.startswith(logMessage_prefix)
        ]
        # Verify logMessage count.
        self.assertEqual(len(logMessage_output), 10)

        message_addr_pattern = r"\b0x[0-9A-Fa-f]+\b"
        message_content = '"Hello from main!"'

        # Verify logMessage match.
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx + 3
            self.assertRegex(
                logMessage_line,
                f"{logMessage_prefix}{result}, {message_addr_pattern} {message_content}",
            )
        session.continue_to_exit()

    @skipIfWindows
    def test_logmessage_advanced(self):
        """Tests breakpoint logmessage functionality for complex expression."""
        session = self.build_and_create_session()
        initial_stop = self.stop_at_before_loop_line(session)
        source = self.getSourcePath("main.cpp")
        before_loop_line = line_number(source, "// break loop")
        after_loop_line = line_number(source, "// after loop")

        # Set two breakpoints:
        # 1. First at the loop line with logMessage
        # 2. Second guard breakpoint at a line after loop
        logMessage_prefix = "This is log message for { -- "
        logMessage = (
            logMessage_prefix
            + "{int y = 0; if (i % 3 == 0) { y = i + 3;} else {y = i * 3;} y}"
        )
        [_, post_loop_breakpoint_id] = session.resolve_source_breakpoints(
            self.main_path,
            [
                SourceBreakpoint(before_loop_line, logMessage=logMessage),
                SourceBreakpoint(after_loop_line),
            ],
        )

        post_loop_stop = session.continue_to_breakpoint(post_loop_breakpoint_id)
        captured = session.collect_console(after=initial_stop, until=post_loop_stop)
        logMessage_output = [
            line
            for line in captured.seen_texts.splitlines()
            if line.startswith(logMessage_prefix)
        ]
        # Verify logMessage count.
        self.assertEqual(len(logMessage_output), 10)

        # Verify logMessage match.
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx + 3 if idx % 3 == 0 else idx * 3
            self.assertEqual(logMessage_line, logMessage_prefix + str(result))

    @skipIfWindows
    def test_logmessage_format(self):
        """Tests breakpoint logmessage functionality with format."""
        session = self.build_and_create_session()
        initial_stop = self.stop_at_before_loop_line(session)
        source = self.getSourcePath("main.cpp")
        loop_line = line_number(source, "// break loop")
        after_loop_line = line_number(source, "// after loop")

        # Set two breakpoints:
        # 1. First at the loop line with logMessage.
        # 2. Second guard breakpoint at a line after loop.
        logMessage_prefix = "This is log message for -- "
        logMessage_with_format = "part1\tpart2\bpart3\x64part4"
        logMessage_with_format_raw = r"part1\tpart2\bpart3\x64part4"
        logMessage = logMessage_prefix + logMessage_with_format_raw + "{i - 1}"
        [_, post_loop_breakpoint_id] = session.resolve_source_breakpoints(
            self.main_path,
            [
                SourceBreakpoint(loop_line, logMessage=logMessage),
                SourceBreakpoint(after_loop_line),
            ],
        )

        post_loop_stop = session.continue_to_breakpoint(post_loop_breakpoint_id)
        captured = session.collect_console(after=initial_stop, until=post_loop_stop)
        logMessage_output = [
            line
            for line in captured.seen_texts.splitlines()
            if line.startswith(logMessage_prefix)
        ]
        # Verify logMessage count.
        self.assertEqual(len(logMessage_output), 10)

        # Verify logMessage match.
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx - 1
            self.assertEqual(
                logMessage_line,
                logMessage_prefix + logMessage_with_format + str(result),
            )

    @skipIfWindows
    def test_logmessage_format_failure(self):
        """Tests breakpoint logmessage format with parsing failure."""
        session = self.build_and_create_session()
        initial_stop = self.stop_at_before_loop_line(session)
        source = self.getSourcePath("main.cpp")
        loop_line = line_number(source, "// break loop")
        after_loop_line = line_number(source, "// after loop")

        # Set two breakpoints:
        # 1. First at the loop line with logMessage.
        # 2. Second guard breakpoint at a line after loop.
        logMessage_prefix = "This is log message for -- "
        # log message missing hex number.
        logMessage = logMessage_prefix + r"part1\x"
        [loop_breakpoint_id, _] = session.resolve_source_breakpoints(
            self.main_path,
            [
                SourceBreakpoint(loop_line, logMessage=logMessage),
                SourceBreakpoint(after_loop_line),
            ],
        )

        # The adapter emits the format error to the console during
        # setBreakpoints and falls back to a real stop at the loop breakpoint
        # when execution resumes.
        loop_stop_event = session.continue_to_breakpoint(loop_breakpoint_id)
        captured = session.collect_console(after=initial_stop, until=loop_stop_event)

        failure_prefix = "Log message has error:"
        logMessage_failure_output = [
            line.strip()
            for line in captured.seen_texts.splitlines()
            if line.startswith(failure_prefix)
        ]
        self.assertEqual(len(logMessage_failure_output), 1)
        self.assertEqual(
            logMessage_failure_output[0],
            failure_prefix + " missing hex number following '\\x'",
        )
