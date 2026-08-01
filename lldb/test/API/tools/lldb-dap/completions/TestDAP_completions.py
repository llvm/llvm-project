"""
Test lldb-dap completions request
"""

# FIXME: remove when LLDB_MINIMUM_PYTHON_VERSION > 3.8
from __future__ import annotations

from typing import Optional, Union

from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import (
    CompletionItem,
    LaunchArgs,
    StoppedReason,
)


session_completion = CompletionItem(
    label="session",
    detail="Commands controlling LLDB session.",
)
settings_completion = CompletionItem(
    label="settings",
    detail="Commands for managing LLDB settings.",
)
memory_completion = CompletionItem(
    label="memory",
    detail="Commands for operating on memory in the current target process.",
)
command_var_completion = CompletionItem(
    label="var",
    detail="Show variables for the current stack frame. Defaults to all arguments and local variables in scope. Names of argument, local, file static and file global variables can be specified.",
    length=3,
)
variable_var_completion = CompletionItem(label="var", detail="vector<baz> &", length=3)
variable_var1_completion = CompletionItem(label="var1", detail="int &")
variable_var2_completion = CompletionItem(label="var2", detail="int &")

str1_completion = CompletionItem(
    label="str1",
    detail="std::string &",
)


# Older version of libcxx produce slightly different typename strings for
# templates like vector.
@skipIf(compiler="clang", compiler_version=["<", "16.0"])
class TestDAP_completions(DAPTestCaseBase):
    def verify_completions(
        self,
        input: str,
        *,
        expected: Union[CompletionItem, set[CompletionItem]],
        not_expected: Optional[Union[CompletionItem, set[CompletionItem]]] = None,
        frame_id: Optional[int] = None,
    ):
        if isinstance(expected, CompletionItem):
            expected = {expected}
        if isinstance(not_expected, CompletionItem):
            not_expected = {not_expected}

        completions = set(self._session.get_completions(input, frameId=frame_id))

        for item in expected:
            self.assertIn(item, completions, f"\nCompletion for input: {input}")

        for item in not_expected or set():
            with self.subTest(f"Not expected completion : {item}"):
                self.assertNotIn(item, completions)

    def setup_debuggee(self):
        """Creates and returns the session and breakpoint stop event."""
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        session = self.build_and_create_session()
        self._session = session
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(
                source,
                [
                    line_number(source, "// breakpoint 1"),
                    line_number(source, "// breakpoint 2"),
                ],
            )
        return session, session.verify_stopped_on_breakpoint(after=ctx.process_event)

    def verify_non_ascii_completion(self, session: DAPTestSession, alias_cmd: str):
        """Creates an command alias for the `next` command and
        verify if it has completion for the command and its help.

        It assumes we are in command mode in the repl.
        """
        session.evaluate(f"command alias {alias_cmd} next", context="repl")

        part = alias_cmd[:2]  # first two characters
        part_codeunits = len(part.encode("utf-16-le")) // 2

        next_detail = "Source level single step, stepping over calls.  Defaults to current thread unless specified."
        expected_item = CompletionItem(
            label=alias_cmd, detail=next_detail, length=part_codeunits
        )

        # Verify the command and it's help.
        self.verify_completions(part, expected=expected_item)
        self.verify_completions(f"help {part}", expected=expected_item)

        session.evaluate(f"command unalias {alias_cmd}", context="repl")

    def test_command_completions(self):
        """Tests completion requests for lldb commands, within "repl-mode=command"."""
        session, _ = self.setup_debuggee()
        stop_event = session.continue_to_next_stop(exp_reason=StoppedReason.BREAKPOINT)

        session.evaluate("`lldb-dap repl-mode command", context="repl")
        top_frame_id = session.top_frame_from(stop_event).frame.id

        # Provides completion for top-level commands.
        self.verify_completions(
            "se",
            expected={
                session_completion.clone(length=2),
                settings_completion.clone(length=2),
            },
            frame_id=top_frame_id,
        )
        # Provides completions for sub-commands.
        self.verify_completions(
            "memory ",
            expected={
                CompletionItem(
                    label="read",
                    detail="Read from the memory of the current target process.",
                ),
                CompletionItem(
                    label="region",
                    detail="Get information on the memory region containing an address "
                    "in the current target process.\nIf this command is given an "
                    "<address-expression> once and then repeated without options, "
                    "it will try to print the memory region that follows the "
                    "previously printed region. The command can be repeated "
                    "until the end of the address range is reached.",
                ),
            },
            frame_id=top_frame_id,
        )

        # Provides completions for parameter values of commands
        self.verify_completions(
            "`log enable  ",
            expected=CompletionItem(label="gdb-remote"),
            frame_id=top_frame_id,
        )

        # Also works if the escape prefix is used
        self.verify_completions(
            "`mem",
            expected=memory_completion.clone(length=3),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "`",
            expected={session_completion, settings_completion, memory_completion},
            frame_id=top_frame_id,
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            'setting "se',
            expected=CompletionItem(
                label="set",
                detail="Set the value of the specified debugger setting.",
                length=3,
            ),
            frame_id=top_frame_id,
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            "'mem",
            expected=memory_completion.clone(length=4),
            frame_id=top_frame_id,
        )

        # Completes expressions with quotes inside
        self.verify_completions(
            'expr " "; typed',
            expected=CompletionItem(label="typedef", length=5),
            frame_id=top_frame_id,
        )

        # Provides completions for commands, but not variables
        self.verify_completions(
            "var",
            expected=command_var_completion,
            not_expected=variable_var_completion,
            frame_id=top_frame_id,
        )

        # Completes partial completion
        self.verify_completions(
            "plugin list ar",
            expected=CompletionItem(label="architecture", length=2),
            frame_id=top_frame_id,
        )

        # Complete custom command with non-ASCII character.
        self.verify_non_ascii_completion(session, "n€xt")  # 3 bytes €
        self.verify_non_ascii_completion(session, "n£xt")  # 2 bytes £
        self.verify_non_ascii_completion(session, "n💩xt")  # 4 bytes 💩
        self.verify_non_ascii_completion(session, "√∂xt")  # starts with non-ASCII
        self.verify_non_ascii_completion(session, "one_seç")  # ends with non-ASCII

    def test_variable_completions(self):
        """Tests completion requests in "repl-mode=variable" """

        session, stop_event = self.setup_debuggee()
        top_frame_id = session.top_frame_from(stop_event).frame.id
        session.evaluate(
            "`lldb-dap repl-mode variable", context="repl", frameId=top_frame_id
        )

        # Provides completions for variables, but not command.
        self.verify_completions(
            "var",
            expected=variable_var_completion,
            not_expected=command_var_completion,
            frame_id=top_frame_id,
        )

        # We stopped inside `fun`, so we shouldn't see variables from main.
        self.verify_completions(
            "var",
            expected=variable_var_completion,
            not_expected={
                variable_var1_completion.clone(length=3),
                variable_var2_completion.clone(length=3),
            },
            frame_id=top_frame_id,
        )

        # We should see global keywords but not variables inside main.
        self.verify_completions(
            "str",
            expected=CompletionItem(label="struct", length=3),
            not_expected=str1_completion.clone(length=3),
            frame_id=top_frame_id,
        )

        stop_event = session.continue_to_next_stop()
        top_frame_id = session.top_frame_from(stop_event).frame.id

        # We stopped in `main`, so we should see variables from main but
        # not from the other function.
        self.verify_completions(
            "var",
            expected={
                variable_var1_completion.clone(length=3),
                variable_var2_completion.clone(length=3),
            },
            not_expected=variable_var_completion.clone(length=3),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "str",
            expected={
                CompletionItem(label="struct", length=3),
                str1_completion.clone(length=3),
            },
            frame_id=top_frame_id,
        )

        self.assertIsNotNone(session.get_completions("ƒ", top_frame_id))
        # Test utf8 after ascii.
        session.get_completions("mƒ", top_frame_id)

        # Completion also works for more complex expressions
        self.verify_completions(
            "foo1.v",
            expected=CompletionItem(label="foo1.var1", detail="int", length=6),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "foo1.my_bar_object.v",
            expected=CompletionItem(
                label="foo1.my_bar_object.var1", detail="int", length=20
            ),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "foo1.var1 + foo1.v",
            expected=CompletionItem(label="foo1.var1", detail="int", length=6),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "foo1.var1 + v",
            expected=CompletionItem(label="var1", detail="int &", length=1),
            frame_id=top_frame_id,
        )

        # should correctly handle spaces between objects and member operators
        self.verify_completions(
            "foo1 .v",
            expected=CompletionItem(label=".var1", detail="int", length=2),
            not_expected=CompletionItem(label=".var2", detail="int", length=2),
            frame_id=top_frame_id,
        )

        self.verify_completions(
            "foo1 . v",
            expected=CompletionItem(label="var1", detail="int", length=1),
            not_expected=CompletionItem(label="var2", detail="int", length=1),
            frame_id=top_frame_id,
        )

        # Even in variable mode, we can still use the escape prefix
        self.verify_completions(
            "`mem",
            expected=memory_completion.clone(length=3),
            frame_id=top_frame_id,
        )

    def test_auto_completions(self):
        """Tests completion requests in "repl-mode=auto"."""
        session, stop_event = self.setup_debuggee()
        session.evaluate("`lldb-dap repl-mode auto", context="repl")
        top_frame_id = session.top_frame_from(stop_event).frame.id

        # Stopped at breakpoint 1
        # 'var' variable is in scope, completions should not show any warning.
        # We check this at the end of the test.
        session.get_completions("var ", top_frame_id)
        stop_event = session.continue_to_next_stop(exp_reason=StoppedReason.BREAKPOINT)

        # We stopped in `main` function. Variables `var1` and `var2` are in scope.
        # Make sure, we offer all completions
        self.verify_completions(
            "va",
            expected={
                command_var_completion.clone(length=2),
                variable_var1_completion.clone(length=2),
                variable_var2_completion.clone(length=2),
            },
            frame_id=top_frame_id,
        )

        # If we are using the escape prefix, only commands are suggested, but no variables
        self.verify_completions(
            "`va",
            expected=command_var_completion.clone(length=2),
            not_expected={
                variable_var1_completion.clone(length=2),
                variable_var2_completion.clone(length=2),
            },
            frame_id=top_frame_id,
        )

        # TODO: Note we are not checking the result because the `expression --` command adds an extra character
        # for non ascii variables.
        self.assertTrue(session.get_completions("ƒ", top_frame_id))

        session.continue_to_exit()
        console_str = session.get_console()
        # we check in console to avoid waiting for output event.
        self.assertNotIn(
            "Expression 'var' is both an LLDB command and variable", console_str
        )
