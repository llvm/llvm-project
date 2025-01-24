"""
Test lldb-dap completions request
"""


import lldbdap_testcase
import dap_server
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

session_completion = {
    "text": "session",
    "label": "session -- Commands controlling LLDB session.",
}
settings_completion = {
    "text": "settings",
    "label": "settings -- Commands for managing LLDB settings.",
}
memory_completion = {
    "text": "memory",
    "label": "memory -- Commands for operating on memory in the current target process.",
}
command_var_completion = {
    "text": "var",
    "label": "var -- Show variables for the current stack frame. Defaults to all arguments and local variables in scope. Names of argument, local, file static and file global variables can be specified.",
}
variable_var_completion = {
    "text": "var",
    "label": "var -- vector<baz> &",
}
variable_var1_completion = {"text": "var1", "label": "var1 -- int &"}
variable_var2_completion = {"text": "var2", "label": "var2 -- int &"}

# Older version of libcxx produce slightly different typename strings for
# templates like vector.
@skipIf(compiler="clang", compiler_version=["<", "16.0"])
class TestDAP_completions(lldbdap_testcase.DAPTestCaseBase):
    def verify_completions(self, actual_list, expected_list, not_expected_list=[]):
        for expected_item in expected_list:
            self.assertIn(expected_item, actual_list)

        for not_expected_item in not_expected_list:
            self.assertNotIn(not_expected_item, actual_list)


    def setup_debugee(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")

        self.set_source_breakpoints(source, [breakpoint1_line, breakpoint2_line])

    def test_command_completions(self):
        """
        Tests completion requests for lldb commands, within "repl-mode=command"
        """
        self.setup_debugee()
        self.continue_to_next_stop()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode command", context="repl"
        )
        self.assertTrue(res["success"])

        # Provides completion for top-level commands
        self.verify_completions(
            self.dap_server.get_completions("se"),
            [session_completion, settings_completion],
        )

        # Provides completions for sub-commands
        self.verify_completions(
            self.dap_server.get_completions("memory "),
            [
                {
                    "text": "read",
                    "label": "read -- Read from the memory of the current target process.",
                },
                {
                    "text": "region",
                    "label": "region -- Get information on the memory region containing an address in the current target process.",
                },
            ],
        )

        # Provides completions for parameter values of commands
        self.verify_completions(
            self.dap_server.get_completions("`log enable  "),
            [{"text": "gdb-remote", "label": "gdb-remote"}],
        )

        # Also works if the escape prefix is used
        self.verify_completions(
            self.dap_server.get_completions("`mem"), [memory_completion]
        )

        self.verify_completions(
            self.dap_server.get_completions("`"),
            [session_completion, settings_completion, memory_completion],
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            self.dap_server.get_completions('setting "se'),
            [
                {
                    "text": "set",
                    "label": "set -- Set the value of the specified debugger setting.",
                }
            ],
        )

        # Completes an incomplete quoted token
        self.verify_completions(
            self.dap_server.get_completions("'mem"),
            [memory_completion],
        )

        # Completes expressions with quotes inside
        self.verify_completions(
            self.dap_server.get_completions('expr " "; typed'),
            [{"text": "typedef", "label": "typedef"}],
        )

        # Provides completions for commands, but not variables
        self.verify_completions(
            self.dap_server.get_completions("var"),
            [command_var_completion],
            [variable_var_completion],
        )

    def test_variable_completions(self):
        """
        Tests completion requests in "repl-mode=variable"
        """
        self.setup_debugee()
        self.continue_to_next_stop()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode variable", context="repl"
        )
        self.assertTrue(res["success"])

        # Provides completions for varibles, but not command
        self.verify_completions(
            self.dap_server.get_completions("var"),
            [variable_var_completion],
            [command_var_completion],
        )

        # We stopped inside `fun`, so we shouldn't see variables from main
        self.verify_completions(
            self.dap_server.get_completions("var"),
            [variable_var_completion],
            [
                variable_var1_completion,
                variable_var2_completion,
            ],
        )

        # We should see global keywords but not variables inside main
        self.verify_completions(
            self.dap_server.get_completions("str"),
            [{"text": "struct", "label": "struct"}],
            [{"text": "str1", "label": "str1 -- std::string &"}],
        )

        self.continue_to_next_stop()

        # We stopped in `main`, so we should see variables from main but
        # not from the other function
        self.verify_completions(
            self.dap_server.get_completions("var"),
            [
                variable_var1_completion,
                variable_var2_completion,
            ],
            [
                variable_var_completion,
            ],
        )

        self.verify_completions(
            self.dap_server.get_completions("str"),
            [
                {"text": "struct", "label": "struct"},
                {"text": "str1", "label": "str1 -- string &"},
            ],
        )

        # Completion also works for more complex expressions
        self.verify_completions(
            self.dap_server.get_completions("foo1.v"),
            [{"text": "var1", "label": "foo1.var1 -- int"}],
        )

        self.verify_completions(
            self.dap_server.get_completions("foo1.my_bar_object.v"),
            [{"text": "var1", "label": "foo1.my_bar_object.var1 -- int"}],
        )

        self.verify_completions(
            self.dap_server.get_completions("foo1.var1 + foo1.v"),
            [{"text": "var1", "label": "foo1.var1 -- int"}],
        )

        self.verify_completions(
            self.dap_server.get_completions("foo1.var1 + v"),
            [{"text": "var1", "label": "var1 -- int &"}],
        )

        # should correctly handle spaces between objects and member operators
        self.verify_completions(
            self.dap_server.get_completions("foo1 .v"),
            [{"text": "var1", "label": ".var1 -- int"}],
            [{"text": "var2", "label": ".var2 -- int"}],
        )

        self.verify_completions(
            self.dap_server.get_completions("foo1 . v"),
            [{"text": "var1", "label": "var1 -- int"}],
            [{"text": "var2", "label": "var2 -- int"}],
        )

        # Even in variable mode, we can still use the escape prefix
        self.verify_completions(
            self.dap_server.get_completions("`mem"), [memory_completion]
        )

    def test_auto_completions(self):
        """
        Tests completion requests in "repl-mode=auto"
        """
        self.setup_debugee()

        res = self.dap_server.request_evaluate(
            "`lldb-dap repl-mode auto", context="repl"
        )
        self.assertTrue(res["success"])

        self.continue_to_next_stop()
        self.continue_to_next_stop()

        # We are stopped inside `main`. Variables `var1` and `var2` are in scope.
        # Make sure, we offer all completions
        self.verify_completions(
            self.dap_server.get_completions("va"),
            [
                command_var_completion,
                variable_var1_completion,
                variable_var2_completion,
            ],
        )

        # If we are using the escape prefix, only commands are suggested, but no variables
        self.verify_completions(
            self.dap_server.get_completions("`va"),
            [
                command_var_completion,
            ],
            [
                variable_var1_completion,
                variable_var2_completion,
            ],
        )
