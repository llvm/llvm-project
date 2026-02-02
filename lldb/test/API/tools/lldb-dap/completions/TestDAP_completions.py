"""
Test lldb-dap completions request
"""

# FIXME: remove when LLDB_MINIMUM_PYTHON_VERSION > 3.8
from __future__ import annotations

import json
from typing import Optional
import lldbdap_testcase
from dataclasses import dataclass, replace, asdict
from lldbsuite.test.decorators import skipIf
from lldbsuite.test.lldbtest import line_number


@dataclass(frozen=True)
class CompletionItem:
    label: str
    text: Optional[str] = None
    detail: Optional[str] = None
    start: Optional[int] = None
    length: int = 0

    def __repr__(self):
        # use json as it easier to see the diff on failure.
        return json.dumps(asdict(self), indent=4)

    def clone(self, **kwargs) -> CompletionItem:
        """Creates a copy of this CompletionItem with specified fields modified."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Scenario:
    input: str
    expected: set[CompletionItem]
    not_expected: Optional[set[CompletionItem]] = None


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
class TestDAP_completions(lldbdap_testcase.DAPTestCaseBase):
    def verify_completions(self, case: Scenario):
        completions = {
            CompletionItem(**comp)
            for comp in self.dap_server.get_completions(case.input)
        }

        # handle expected completions
        for exp_comp in case.expected:
            self.assertIn(
                exp_comp, completions, f"\nCompletion for input: {case.input}"
            )

        # unexpected completions
        for not_exp_comp in case.not_expected or set():
            with self.subTest(f"Not expected completion : {not_exp_comp}"):
                self.assertNotIn(not_exp_comp, completions)

    def setup_debuggee(self):
        program = self.getBuildArtifact("a.out")
        source = "main.cpp"
        self.build_and_launch(program)
        self.set_source_breakpoints(
            source,
            [
                line_number(source, "// breakpoint 1"),
                line_number(source, "// breakpoint 2"),
            ],
        )

    def verify_non_ascii_completion(self, alias_cmd: str):
        """Creates an command alias for the `next` command and
        verify if it has completion for the command and its help.

        It assumes we are in command mode in the repl.
        """
        res = self.dap_server.request_evaluate(
            f"command alias {alias_cmd} next", context="repl"
        )
        self.assertTrue(res["success"])

        part = alias_cmd[:2]  # first two characters
        part_codeunits = len(part.encode("utf-16-le")) // 2

        next_detail = "Source level single step, stepping over calls.  Defaults to current thread unless specified."
        expected_item = CompletionItem(
            label=alias_cmd, detail=next_detail, length=part_codeunits
        )

        # complete the command
        self.verify_completions(Scenario(input=part, expected={expected_item}))
        # complete the help
        self.verify_completions(
            Scenario(input=f"help {part}", expected={expected_item})
        )

        # remove the alias
        res = self.dap_server.request_evaluate(
            f"command unalias {alias_cmd}", context="repl"
        )
        self.assertTrue(res["success"])

    def test_command_completions(self):
        """
        Tests completion requests for lldb commands, within "repl-mode=command"
        """
        self.setup_debuggee()
        self.continue_to_next_stop()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode command", context="repl"
        )
        self.assertTrue(res["success"])

        # Provides completion for top-level commands
        self.verify_completions(
            Scenario(
                input="se",
                expected={
                    session_completion.clone(length=2),
                    settings_completion.clone(length=2),
                },
            )
        )
        # Provides completions for sub-commands
        self.verify_completions(
            Scenario(
                input="memory ",
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
            ),
        )

        # Provides completions for parameter values of commands
        self.verify_completions(
            Scenario(
                input="`log enable  ", expected={CompletionItem(label="gdb-remote")}
            )
        )

        # Also works if the escape prefix is used
        self.verify_completions(
            Scenario(input="`mem", expected={memory_completion.clone(length=3)})
        )

        self.verify_completions(
            Scenario(
                input="`",
                expected={session_completion, settings_completion, memory_completion},
            )
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            Scenario(
                input='setting "se',
                expected={
                    CompletionItem(
                        label="set",
                        detail="Set the value of the specified debugger setting.",
                        length=3,
                    )
                },
            )
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            Scenario(input="'mem", expected={memory_completion.clone(length=4)})
        )

        # Completes expressions with quotes inside
        self.verify_completions(
            Scenario(
                input='expr " "; typed',
                expected={CompletionItem(label="typedef", length=5)},
            )
        )

        # Provides completions for commands, but not variables
        self.verify_completions(
            Scenario(
                input="var",
                expected={command_var_completion},
                not_expected={variable_var_completion},
            )
        )

        # Completes partial completion
        self.verify_completions(
            Scenario(
                input="plugin list ar",
                expected={CompletionItem(label="architecture", length=2)},
            )
        )

        # Complete custom command with non ascii character.
        self.verify_non_ascii_completion("n€xt")  # 2 bytes £
        self.verify_non_ascii_completion("n£xt")  # 3 bytes €
        self.verify_non_ascii_completion("n💩xt")  # 4 bytes 💩
        self.verify_non_ascii_completion("√∂xt")  # start with non ascii
        self.verify_non_ascii_completion("one_seç")  # ends with non ascii

    def test_variable_completions(self):
        """
        Tests completion requests in "repl-mode=variable"
        """
        self.setup_debuggee()
        self.continue_to_next_stop()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode variable", context="repl"
        )
        self.assertTrue(res["success"])

        # Provides completions for varibles, but not command
        self.verify_completions(
            Scenario(
                input="var",
                expected={variable_var_completion},
                not_expected={command_var_completion},
            )
        )

        # We stopped inside `fun`, so we shouldn't see variables from main
        self.verify_completions(
            Scenario(
                input="var",
                expected={variable_var_completion},
                not_expected={
                    variable_var1_completion.clone(length=3),
                    variable_var2_completion.clone(length=3),
                },
            )
        )

        # We should see global keywords but not variables inside main
        self.verify_completions(
            Scenario(
                input="str",
                expected={CompletionItem(label="struct", length=3)},
                not_expected={str1_completion.clone(length=3)},
            )
        )

        self.continue_to_next_stop()

        # We stopped in `main`, so we should see variables from main but
        # not from the other function
        self.verify_completions(
            Scenario(
                input="var",
                expected={
                    variable_var1_completion.clone(length=3),
                    variable_var2_completion.clone(length=3),
                },
                not_expected={
                    variable_var_completion.clone(length=3),
                },
            )
        )

        self.verify_completions(
            Scenario(
                input="str",
                expected={
                    CompletionItem(label="struct", length=3),
                    str1_completion.clone(length=3),
                },
            )
        )

        self.assertIsNotNone(self.dap_server.get_completions("ƒ"))
        # Test utf8 after ascii.
        # TODO
        self.dap_server.get_completions("mƒ")

        # Completion also works for more complex expressions
        self.verify_completions(
            Scenario(
                input="foo1.v",
                expected={CompletionItem(label="foo1.var1", detail="int", length=6)},
            )
        )

        self.verify_completions(
            Scenario(
                input="foo1.my_bar_object.v",
                expected={
                    CompletionItem(
                        label="foo1.my_bar_object.var1", detail="int", length=20
                    )
                },
            )
        )

        self.verify_completions(
            Scenario(
                input="foo1.var1 + foo1.v",
                expected={CompletionItem(label="foo1.var1", detail="int", length=6)},
            )
        )

        self.verify_completions(
            Scenario(
                input="foo1.var1 + v",
                expected={CompletionItem(label="var1", detail="int &", length=1)},
            )
        )

        # should correctly handle spaces between objects and member operators
        self.verify_completions(
            Scenario(
                input="foo1 .v",
                expected={CompletionItem(label=".var1", detail="int", length=2)},
                not_expected={CompletionItem(label=".var2", detail="int", length=2)},
            )
        )

        self.verify_completions(
            Scenario(
                input="foo1 . v",
                expected={CompletionItem(label="var1", detail="int", length=1)},
                not_expected={CompletionItem(label="var2", detail="int", length=1)},
            )
        )

        # Even in variable mode, we can still use the escape prefix
        self.verify_completions(
            Scenario(input="`mem", expected={memory_completion.clone(length=3)})
        )

    def test_auto_completions(self):
        """
        Tests completion requests in "repl-mode=auto"
        """
        self.setup_debuggee()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode auto", context="repl"
        )
        self.assertTrue(res["success"])

        self.continue_to_next_stop()

        # Stopped at breakpoint 1
        # 'var' variable is in scope, completions should not show any warning.
        self.dap_server.get_completions("var ")
        self.continue_to_next_stop()

        # We are stopped inside `main`. Variables `var1` and `var2` are in scope.
        # Make sure, we offer all completions
        self.verify_completions(
            Scenario(
                input="va",
                expected={
                    command_var_completion.clone(length=2),
                    variable_var1_completion.clone(length=2),
                    variable_var2_completion.clone(length=2),
                },
            )
        )

        # If we are using the escape prefix, only commands are suggested, but no variables
        self.verify_completions(
            Scenario(
                input="`va",
                expected={
                    command_var_completion.clone(length=2),
                },
                not_expected={
                    variable_var1_completion.clone(length=2),
                    variable_var2_completion.clone(length=2),
                },
            )
        )

        # TODO: Note we are not checking the result because the `expression --` command adds an extra character
        # for non ascii variables.
        self.assertIsNotNone(self.dap_server.get_completions("ƒ"))

        self.continue_to_exit()
        console_str = self.get_console()
        # we check in console to avoid waiting for output event.
        self.assertNotIn(
            "Expression 'var' is both an LLDB command and variable", console_str
        )
