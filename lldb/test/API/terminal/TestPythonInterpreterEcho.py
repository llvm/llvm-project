"""
Test that typing python expression in the terminal is echoed back to stdout.
"""

from lldbsuite.test.decorators import skipIfAsan
from lldbsuite.test.lldbpexpect import PExpectTest


@skipIfAsan
class PythonInterpreterEchoTest(PExpectTest):
    PYTHON_PROMPT = ">>> "

    def verify_command_echo(
        self, command: str, expected_output: str = "", is_regex: bool = False
    ):
        assert self.child != None
        child = self.child
        self.assertIsNotNone(self.child, "expected a running lldb process.")

        child.sendline(command)

        # Build pattern list: match whichever comes first (output or prompt).
        # This prevents waiting for a timeout if there's no match.
        pattern = []
        match_expected = expected_output and len(expected_output) > 0

        if match_expected:
            pattern.append(expected_output)
        pattern.append(self.PYTHON_PROMPT)

        expect_func = child.expect if is_regex else child.expect_exact
        match_idx = expect_func(pattern)
        if match_expected:
            self.assertEqual(
                match_idx, 0, "Expected output `{expected_output}` in stdout."
            )

        self.assertIsNotNone(self.child.before, "Expected output before prompt")
        self.assertIsInstance(self.child.before, bytes)
        echoed_text: str = self.child.before.decode("ascii").strip()
        self.assertEqual(
            command, echoed_text, f"Command '{command}' should be echoed to stdout."
        )

        if match_expected:
            child.expect_exact(self.PYTHON_PROMPT)

    def test_python_interpreter_echo(self):
        """Test that that the user typed commands is echoed to stdout"""

        self.launch(use_colors=False, dimensions=(100, 100))

        # Enter the python interpreter.
        self.verify_command_echo(
            "script --language python --", expected_output="Python.*\\.", is_regex=True
        )
        self.child_in_script_interpreter = True

        self.verify_command_echo("val = 300")
        self.verify_command_echo(
            "print('result =', 300)", expected_output="result = 300"
        )
