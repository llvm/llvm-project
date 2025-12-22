"""
Test that target var can resolve complex DWARF expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class targetCommandTestCase(TestBase):
    @skipIfDarwinEmbedded  # needs x86_64
    @skipIf(debug_info="gmodules")  # not relevant
    @skipIf(compiler="clang", compiler_version=["<", "7.0"])
    def testTargetVarExpr(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self.expect(
            "help target variable",
            substrs=[
                "--no-args",
                "--no-recognized-args",
                "--no-locals",
                "--show-globals",
            ],
            matching=False,
        )
        self.expect("target variable i", substrs=["i", "42"])
        self.expect(
            "target variable var", patterns=[r"\(incomplete \*\) var = 0[xX](0)*dead"]
        )
        self.expect(
            "target variable var[0]",
            error=True,
            substrs=["can't find global variable 'var[0]'"],
        )

        command_result = lldb.SBCommandReturnObject()
        result = self.ci.HandleCommand("target var", command_result)
        value_list = command_result.GetValues(lldb.eNoDynamicValues)
        self.assertGreaterEqual(value_list.GetSize(), 2)
        value_names = []
        for value in value_list:
            value_names.append(value.GetName())
        self.assertIn("i", value_names)
