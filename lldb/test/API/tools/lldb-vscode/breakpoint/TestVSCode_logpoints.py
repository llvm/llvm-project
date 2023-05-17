"""
Test lldb-vscode logpoints feature.
"""


import vscode
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_logpoints(lldbvscode_testcase.VSCodeTestCaseBase):

    def setUp(self):
        lldbvscode_testcase.VSCodeTestCaseBase.setUp(self)

        self.main_basename = 'main-copy.cpp'
        self.main_path = os.path.realpath(self.getBuildArtifact(self.main_basename))

    @skipIfWindows
    @skipIfRemote
    def test_logmessage_basic(self):
        '''Tests breakpoint logmessage basic functionality.'''
        before_loop_line = line_number('main.cpp', '// before loop')
        loop_line = line_number('main.cpp', '// break loop')
        after_loop_line = line_number('main.cpp', '// after loop')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set a breakpoint at a line before loop
        before_loop_breakpoint_ids = self.set_source_breakpoints(
            self.main_path,
            [before_loop_line])
        self.assertEquals(len(before_loop_breakpoint_ids), 1, "expect one breakpoint")

        self.vscode.request_continue()

        # Verify we hit the breakpoint before loop line
        self.verify_breakpoint_hit(before_loop_breakpoint_ids)

        # Swallow old console output
        self.get_console()

        # Set two breakpoints:
        # 1. First at the loop line with logMessage
        # 2. Second guard breakpoint at a line after loop
        logMessage_prefix = "This is log message for { -- "
        logMessage = logMessage_prefix + "{i + 3}"
        [loop_breakpoint_id, post_loop_breakpoint_id] = self.set_source_breakpoints(
            self.main_path,
            [loop_line, after_loop_line],
            [{'logMessage': logMessage}, {}]
        )

        # Continue to trigger the breakpoint with log messages
        self.vscode.request_continue()

        # Verify we hit the breakpoint after loop line
        self.verify_breakpoint_hit([post_loop_breakpoint_id])

        output = self.get_console()
        lines = output.splitlines()
        logMessage_output = []
        for line in lines:
            if line.startswith(logMessage_prefix):
                logMessage_output.append(line)

        # Verify logMessage count
        loop_count = 10
        self.assertEqual(len(logMessage_output), loop_count)

        # Verify log message match
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx + 3
            self.assertEqual(logMessage_line, logMessage_prefix + str(result))


    @skipIfWindows
    @skipIfRemote
    def test_logmessage_advanced(self):
        '''Tests breakpoint logmessage functionality for complex expression.'''
        before_loop_line = line_number('main.cpp', '// before loop')
        loop_line = line_number('main.cpp', '// break loop')
        after_loop_line = line_number('main.cpp', '// after loop')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set a breakpoint at a line before loop
        before_loop_breakpoint_ids = self.set_source_breakpoints(
            self.main_path,
            [before_loop_line])
        self.assertEquals(len(before_loop_breakpoint_ids), 1, "expect one breakpoint")

        self.vscode.request_continue()

        # Verify we hit the breakpoint before loop line
        self.verify_breakpoint_hit(before_loop_breakpoint_ids)

        # Swallow old console output
        self.get_console()

        # Set two breakpoints:
        # 1. First at the loop line with logMessage
        # 2. Second guard breakpoint at a line after loop
        logMessage_prefix = "This is log message for { -- "
        logMessage = logMessage_prefix + "{int y = 0; if (i % 3 == 0) { y = i + 3;} else {y = i * 3;} y}"
        [loop_breakpoint_id, post_loop_breakpoint_id] = self.set_source_breakpoints(
            self.main_path,
            [loop_line, after_loop_line],
            [{'logMessage': logMessage}, {}]
        )

        # Continue to trigger the breakpoint with log messages
        self.vscode.request_continue()

        # Verify we hit the breakpoint after loop line
        self.verify_breakpoint_hit([post_loop_breakpoint_id])

        output = self.get_console()
        lines = output.splitlines()
        logMessage_output = []
        for line in lines:
            if line.startswith(logMessage_prefix):
                logMessage_output.append(line)

        # Verify logMessage count
        loop_count = 10
        self.assertEqual(len(logMessage_output), loop_count)

        # Verify log message match
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx + 3 if idx % 3 == 0 else idx * 3
            self.assertEqual(logMessage_line, logMessage_prefix + str(result))

    @skipIfWindows
    @skipIfRemote
    def test_logmessage_format(self):
        '''
            Tests breakpoint logmessage functionality with format.
        '''
        before_loop_line = line_number('main.cpp', '// before loop')
        loop_line = line_number('main.cpp', '// break loop')
        after_loop_line = line_number('main.cpp', '// after loop')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set a breakpoint at a line before loop
        before_loop_breakpoint_ids = self.set_source_breakpoints(
            self.main_path,
            [before_loop_line])
        self.assertEquals(len(before_loop_breakpoint_ids), 1, "expect one breakpoint")

        self.vscode.request_continue()

        # Verify we hit the breakpoint before loop line
        self.verify_breakpoint_hit(before_loop_breakpoint_ids)

        # Swallow old console output
        self.get_console()

        # Set two breakpoints:
        # 1. First at the loop line with logMessage
        # 2. Second guard breakpoint at a line after loop
        logMessage_prefix = "This is log message for -- "
        logMessage_with_format = "part1\tpart2\bpart3\x64part4"
        logMessage_with_format_raw = r"part1\tpart2\bpart3\x64part4"
        logMessage = logMessage_prefix + logMessage_with_format_raw + "{i - 1}"
        [loop_breakpoint_id, post_loop_breakpoint_id] = self.set_source_breakpoints(
            self.main_path,
            [loop_line, after_loop_line],
            [{'logMessage': logMessage}, {}]
        )

        # Continue to trigger the breakpoint with log messages
        self.vscode.request_continue()

        # Verify we hit the breakpoint after loop line
        self.verify_breakpoint_hit([post_loop_breakpoint_id])

        output = self.get_console()
        lines = output.splitlines()
        logMessage_output = []
        for line in lines:
            if line.startswith(logMessage_prefix):
                logMessage_output.append(line)

        # Verify logMessage count
        loop_count = 10
        self.assertEqual(len(logMessage_output), loop_count)

        # Verify log message match
        for idx, logMessage_line in enumerate(logMessage_output):
            result = idx - 1
            self.assertEqual(logMessage_line, logMessage_prefix + logMessage_with_format + str(result))


    @skipIfWindows
    @skipIfRemote
    def test_logmessage_format_failure(self):
        '''
            Tests breakpoint logmessage format with parsing failure.
        '''
        before_loop_line = line_number('main.cpp', '// before loop')
        loop_line = line_number('main.cpp', '// break loop')
        after_loop_line = line_number('main.cpp', '// after loop')

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set a breakpoint at a line before loop
        before_loop_breakpoint_ids = self.set_source_breakpoints(
            self.main_path,
            [before_loop_line])
        self.assertEquals(len(before_loop_breakpoint_ids), 1, "expect one breakpoint")

        self.vscode.request_continue()

        # Verify we hit the breakpoint before loop line
        self.verify_breakpoint_hit(before_loop_breakpoint_ids)

        # Swallow old console output
        self.get_console()

        # Set two breakpoints:
        # 1. First at the loop line with logMessage
        # 2. Second guard breakpoint at a line after loop
        logMessage_prefix = "This is log message for -- "
        # log message missing hex number.
        logMessage_with_format_raw = r"part1\x"
        logMessage = logMessage_prefix + logMessage_with_format_raw
        [loop_breakpoint_id, post_loop_breakpoint_id] = self.set_source_breakpoints(
            self.main_path,
            [loop_line, after_loop_line],
            [{'logMessage': logMessage}, {}]
        )

        # Continue to trigger the breakpoint with log messages
        self.vscode.request_continue()

        # Verify we hit logpoint breakpoint if it's format has error.
        self.verify_breakpoint_hit([loop_breakpoint_id])

        output = self.get_console()
        lines = output.splitlines()

        failure_prefix = "Log message has error:"
        logMessage_output = []
        logMessage_failure_output = []
        for line in lines:
            if line.startswith(logMessage_prefix):
                logMessage_output.append(line)
            elif line.startswith(failure_prefix):
                logMessage_failure_output.append(line)

        # Verify logMessage failure message
        self.assertEqual(len(logMessage_failure_output), 1)
        self.assertEqual(
            logMessage_failure_output[0].strip(),
            failure_prefix + " missing hex number following '\\x'")
