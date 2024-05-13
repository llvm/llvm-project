import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class InvalidArgsCommandTestCase(TestBase):
    @no_debug_info_test
    def test_script_add(self):
        self.expect(
            "command script add 1 2",
            error=True,
            substrs=["Path component: '1' not found"],
        )

        self.expect(
            "command script add",
            error=True,
            substrs=["'command script add' requires at least one argument"],
        )

    @no_debug_info_test
    def test_script_clear(self):
        self.expect(
            "command script clear f",
            error=True,
            substrs=["'command script clear' doesn't take any arguments"],
        )

    @no_debug_info_test
    def test_script_list(self):
        self.expect(
            "command script list f",
            error=True,
            substrs=["'command script list' doesn't take any arguments"],
        )

    @no_debug_info_test
    def test_script_import(self):
        self.expect(
            "command script import",
            error=True,
            substrs=["command script import needs one or more arguments"],
        )

    @no_debug_info_test
    def test_alias(self):
        self.expect(
            "command alias",
            error=True,
            substrs=["'command alias' requires at least two arguments"],
        )

        self.expect(
            "command alias blub foo",
            error=True,
            substrs=[
                "error: invalid command given to 'command alias'. 'foo' does not begin with a valid command.  No alias created."
            ],
        )

    @no_debug_info_test
    def test_unalias(self):
        self.expect(
            "command unalias",
            error=True,
            substrs=["must call 'unalias' with a valid alias"],
        )

    @no_debug_info_test
    def test_delete(self):
        self.expect(
            "command delete",
            error=True,
            substrs=["must call 'command delete' with one or more valid user"],
        )

    @no_debug_info_test
    def test_regex(self):
        self.expect(
            "command regex",
            error=True,
            substrs=["usage: 'command regex <command-name> "],
        )

    @no_debug_info_test
    def test_source(self):
        self.expect(
            "command source",
            error=True,
            substrs=[
                "'command source' takes exactly one executable filename argument."
            ],
        )
