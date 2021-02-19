"""
Test top-level expressions.
"""



import unittest2

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TopLevelExpressionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Set breakpoint here')
        self.dummy_line = line_number('dummy.cpp',
                                      '// Set breakpoint here')

        # Disable confirmation prompt to avoid infinite wait
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def build_and_run(self):
        """Test top-level expressions."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

    def run_dummy(self):
        self.runCmd("file " + self.getBuildArtifact("dummy"),
                    CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "dummy.cpp",
            self.dummy_line,
            num_expected_locations=1,
            loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

    @add_test_categories(['pyapi'])
    @skipIf(debug_info="gmodules")  # not relevant
    @skipIf(oslist=["windows"])  # Error in record layout on Windows
    def test_top_level_expressions(self):
        self.build_and_run()

        resultFromCode = self.frame().EvaluateExpression("doTest()").GetValueAsUnsigned()

        self.runCmd("kill")

        self.run_dummy()

        codeFile = open('test.cpp', 'r')

        expressions = []
        current_expression = ""

        for line in codeFile:
            if line.startswith("// --"):
                expressions.append(current_expression)
                current_expression = ""
            else:
                current_expression += line

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        options.SetTopLevel(True)

        for expression in expressions:
            self.frame().EvaluateExpression(expression, options)

        resultFromTopLevel = self.frame().EvaluateExpression("doTest()")

        self.assertTrue(resultFromTopLevel.IsValid())
        self.assertEqual(
            resultFromCode,
            resultFromTopLevel.GetValueAsUnsigned())

    def test_top_level_expression_without_target(self):
        self.expect("expr --top-level -- void func() {}", error=True,
                    substrs=["Top-level code needs to be inserted into a runnable target"])

        # FIXME: This doesn't actually generate any code, so LLDB should probably
        # allow these expressions.
        self.expect("expr --top-level -- template<typename T> struct StructT { T m; };", error=True,
                    substrs=["Top-level code needs to be inserted into a runnable target"])
        self.expect("expr --top-level -- struct Struct { int i; };", error=True,
                    substrs=["Top-level code needs to be inserted into a runnable target"])
