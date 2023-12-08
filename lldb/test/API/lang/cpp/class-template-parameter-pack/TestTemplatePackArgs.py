"""
Test that the type of arguments to C++ template classes that have variadic
parameters can be enumerated.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TemplatePackArgsTestCase(TestBase):
    def test_template_argument_pack(self):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "breakpoint here", lldb.SBFileSpec("main.cpp"), exe_name="a.out"
        )
        frame = thread.GetSelectedFrame()

        empty_pack = frame.FindVariable("emptyPack")
        self.assertTrue(
            empty_pack.IsValid(), "make sure we find the emptyPack variable"
        )

        only_pack = frame.FindVariable("onlyPack")
        self.assertTrue(only_pack.IsValid(), "make sure we find the onlyPack variable")
        self.assertEqual(only_pack.GetType().GetNumberOfTemplateArguments(), 4)
        self.assertEqual(
            only_pack.GetType().GetTemplateArgumentType(0).GetName(), "int"
        )
        self.assertEqual(
            only_pack.GetType().GetTemplateArgumentType(1).GetName(), "char"
        )
        self.assertEqual(
            only_pack.GetType().GetTemplateArgumentType(2).GetName(), "double"
        )
        # Access the C<double, 42> template parameter.
        nested_template = only_pack.GetType().GetTemplateArgumentType(3)
        self.assertEqual(nested_template.GetName(), "D<int, int, bool>")
        self.assertEqual(nested_template.GetNumberOfTemplateArguments(), 3)
        self.assertEqual(nested_template.GetTemplateArgumentType(0).GetName(), "int")
        self.assertEqual(nested_template.GetTemplateArgumentType(1).GetName(), "int")
        self.assertEqual(nested_template.GetTemplateArgumentType(2).GetName(), "bool")
