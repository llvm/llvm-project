"""
Test debug-info parsing of synthesized Objective-C properties.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSynthesizedPropertyAccessor(TestBase):
    def test(self):
        self.build()

        (target, _, _, _) = lldbutil.run_to_source_breakpoint(
            self, "return f.fooProp", lldb.SBFileSpec("main.m")
        )

        getters = target.FindFunctions("-[Foo fooProp]", lldb.eFunctionNameTypeSelector)
        self.assertEqual(len(getters), 1)
        getter = getters[0].function.GetType()
        self.assertTrue(getter)
        self.assertEqual(getter.GetDisplayTypeName(), "int ()")

        setters = target.FindFunctions(
            "-[Foo setFooProp:]", lldb.eFunctionNameTypeSelector
        )
        self.assertEqual(len(setters), 1)
        setter = setters[0].function.GetType()
        self.assertTrue(setter)
        self.assertEqual(setter.GetDisplayTypeName(), "void (int)")
