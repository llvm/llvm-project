"""
Test SubstTemplateTypeParam types which are produced as type sugar
when template type parameters are used for example as field types.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_typedef(self):
        target = self.dbg.GetDummyTarget()

        # Declare a template class with a field that uses the template type
        # parameter.
        opts = lldb.SBExpressionOptions()
        opts.SetTopLevel(True)
        result = target.EvaluateExpression("template <typename T> struct X { T f; };", opts)
        # FIXME: This fails with "Couldn't find $__lldb_expr() in the module"
        # but it should succeed. The fact that this code has nothing to run
        # shouldn't be an error.
        # self.assertSuccess(result.GetError())

        # Instantiate and produce a value with that template as the type.
        # The field in the value will have a SubstTemplateTypeParam that
        # should behave like a normal field.
        result = target.EvaluateExpression("X<int> x; x.f = 123; x")
        self.assertEqual(result.GetNumChildren(), 1)
        self.assertEqual(result.GetChildAtIndex(0).GetTypeName(), "int")
        self.assertEqual(result.GetChildAtIndex(0).GetValue(), "123")
