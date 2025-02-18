import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

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

        # Template parameter isn't a NTTP.
        self.assertFalse(template_type.GetTemplateArgumentValue(target, 0))

        # Template parameter index out-of-bounds.
        self.assertFalse(template_type.GetTemplateArgumentValue(target, 2))

        # Template parameter is a NTTP.
        param_val = template_type.GetTemplateArgumentValue(target, 1)
        self.assertEqual(param_val.GetTypeName(), "unsigned int")
        self.assertEqual(param_val.GetValueAsUnsigned(), 2)

        # Try to get an invalid template argument.
        self.assertEqual(
            template_type.GetTemplateArgumentKind(2), lldb.eTemplateArgumentKindNull
        )
        self.assertEqual(template_type.GetTemplateArgumentType(2).GetName(), "")

        value = self.expect_expr("temp2", result_type="Foo<short, -2>")

        # Can't get template parameter value with invalid target.
        self.assertFalse(value.GetType().GetTemplateArgumentValue(lldb.SBTarget(), 1))

        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertTrue(template_param_value)
        self.assertEqual(template_param_value.GetTypeName(), "short")
        self.assertEqual(template_param_value.GetValueAsSigned(), -2)

        value = self.expect_expr("temp3", result_type="Foo<char, 'v'>")
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertTrue(template_param_value)
        self.assertEqual(template_param_value.GetTypeName(), "char")
        self.assertEqual(chr(template_param_value.GetValueAsSigned()), "v")

        # FIXME: type should be Foo<float, 2.0f>
        # FIXME: double/float NTTP parameter values currently not supported.
        value = self.expect_expr("temp4", result_type="Foo<float, float>")
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertFalse(template_param_value)
