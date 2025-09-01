import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    @skipIf(compiler="clang", compiler_version=["<", "20.0"])
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

        value = self.expect_expr("temp4", result_type="Foo<float, 2.000000e+00>")
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertEqual(template_param_value.GetTypeName(), "float")
        # FIXME: this should return a float
        self.assertEqual(template_param_value.GetValueAsSigned(), 2)

        value = self.expect_expr("temp5", result_type="Foo<double, -2.505000e+02>")
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertEqual(template_param_value.GetTypeName(), "double")
        # FIXME: this should return a float
        self.assertEqual(template_param_value.GetValueAsSigned(), -250)

        # FIXME: type should be Foo<int *, &temp1.member>
        value = self.expect_expr("temp6", result_type="Foo<int *, int *>")
        self.assertFalse(value.GetType().GetTemplateArgumentValue(target, 1))

        # FIXME: support wider range of floating point types
        value = self.expect_expr("temp7", result_type="Foo<__fp16, __fp16>")
        self.assertFalse(value.GetType().GetTemplateArgumentValue(target, 1))

        value = self.expect_expr("temp8", result_type="Foo<__fp16, __fp16>")
        self.assertFalse(value.GetType().GetTemplateArgumentValue(target, 1))

        value = self.expect_expr("temp9", result_type="Bar<double, 1.200000e+00>")
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertEqual(template_param_value.GetTypeName(), "double")
        # FIXME: this should return a float
        self.assertEqual(template_param_value.GetValueAsSigned(), 1)

        value = self.expect_expr(
            "temp10", result_type="Bar<float, 1.000000e+00, 2.000000e+00>"
        )
        template_param_value = value.GetType().GetTemplateArgumentValue(target, 1)
        self.assertEqual(template_param_value.GetTypeName(), "float")
        # FIXME: this should return a float
        self.assertEqual(template_param_value.GetValueAsSigned(), 1)

        template_param_value = value.GetType().GetTemplateArgumentValue(target, 2)
        self.assertEqual(template_param_value.GetTypeName(), "float")
        # FIXME: this should return a float
        self.assertEqual(template_param_value.GetValueAsSigned(), 2)
