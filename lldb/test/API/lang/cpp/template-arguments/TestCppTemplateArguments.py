import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        value = self.expect_expr("temp1", result_type="C<int, 2>")
        template_type = value.GetType()
        self.assertEqual(template_type.GetNumberOfTemplateArguments(), 2)

        # Check a type argument.
        self.assertEqual(
            template_type.GetTemplateArgumentKind(0), lldb.eTemplateArgumentKindType
        )
        self.assertEqual(template_type.GetTemplateArgumentType(0).GetName(), "int")

        # Check a integral argument.
        self.assertEqual(
            template_type.GetTemplateArgumentKind(1), lldb.eTemplateArgumentKindIntegral
        )
        self.assertEqual(
            template_type.GetTemplateArgumentType(1).GetName(), "unsigned int"
        )
        # FIXME: There is no way to get the actual value of the parameter.

        # Try to get an invalid template argument.
        self.assertEqual(
            template_type.GetTemplateArgumentKind(2), lldb.eTemplateArgumentKindNull
        )
        self.assertEqual(template_type.GetTemplateArgumentType(2).GetName(), "")
