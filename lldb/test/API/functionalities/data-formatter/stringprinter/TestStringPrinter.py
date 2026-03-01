import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStringPrinter(TestBase):
    def test(self):
        self.build()

        self.addTearDownHook(
            lambda x: x.runCmd("setting set escape-non-printables true")
        )

        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_var_path(
            "charwithtabs",
            summary='"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"',
        )
        self.expect_var_path("a.data", summary='"FOOB"')
        self.expect_var_path("b.data", summary=r'"FO\0B"')
        self.expect_var_path("c.data", summary=r'"F\0O"')
        self.expect_var_path("manytrailingnuls", summary=r'"F\0OO\0BA\0R"')

        for c in ["", "const"]:
            for v in ["", "volatile"]:
                for s in ["", "unsigned"]:
                    summary = '"' + c + v + s + 'char"'
                    self.expect_var_path(c + v + s + "chararray", summary=summary)
                    # These should be printed normally
                    self.expect_var_path(c + v + s + "charstar", summary=summary)

        Schar5 = self.expect_var_path(
            "Schar5", children=[ValueCheck(name="x", value="0")]
        )
        self.assertIsNone(Schar5.GetSummary())
        Scharstar = self.expect_var_path(
            "Scharstar", children=[ValueCheck(name="x", value="0")]
        )
        self.assertIsNone(Scharstar.GetSummary())

        self.runCmd("setting set escape-non-printables false")
        self.expect_var_path(
            "charwithtabs", summary='"Hello\t\tWorld\nI am here\t\tto say hello\n"'
        )
        self.assertTrue(
            self.frame().FindVariable("longconstcharstar").GetSummary().endswith('"...')
        )

        # FIXME: make "b.data" and "c.data" work sanely

        self.expect("frame variable ref", substrs=['(&ref = "Hello")'])
        self.expect_var_path(
            "ref",
            summary=None,
            children=[ValueCheck(name="&ref", summary='"Hello"')],
        )

        # FIXME: should LLDB use "&&refref" for the name here?
        self.expect("frame variable refref", substrs=['(&refref = "Hi")'])
        self.expect_var_path(
            "refref",
            summary=None,
            children=[ValueCheck(name="&refref", summary='"Hi"')],
        )
