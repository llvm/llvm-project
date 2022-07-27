"""
Tests expression evaluation in context of an object.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ContextObjectTestCase(TestBase):

    def test_context_object(self):
        """Tests expression evaluation in context of an object."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, '// Break here', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        #
        # Test C++ struct variable and reference-to-struct variable
        #
        for obj in "cpp_struct", "cpp_struct_ref":
            obj_val = frame.FindVariable(obj)
            self.assertTrue(obj_val.IsValid())

            # Test an empty expression evaluation
            value = obj_val.EvaluateExpression("")
            self.assertFalse(value.IsValid())
            self.assertFalse(value.GetError().Success())

            # Test retrieveing of a field (not a local with the same name)
            value = obj_val.EvaluateExpression("field")
            self.assertTrue(value.IsValid())
            self.assertSuccess(value.GetError())
            self.assertEqual(value.GetValueAsSigned(), 1111)

            # Test functions evaluation
            value = obj_val.EvaluateExpression("function()")
            self.assertTrue(value.IsValid())
            self.assertSuccess(value.GetError())
            self.assertEqual(value.GetValueAsSigned(), 2222)

            # Test that we retrieve the right global
            value = obj_val.EvaluateExpression("global.field")
            self.assertTrue(value.IsValid())
            self.assertSuccess(value.GetError())
            self.assertEqual(value.GetValueAsSigned(), 1111)

        #
        # Test C++ union variable
        #

        obj_val = frame.FindVariable("cpp_union")
        self.assertTrue(obj_val.IsValid())

        # Test retrieveing of a field
        value = obj_val.EvaluateExpression("field_int")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 5555)

        #
        # Test C++ scalar
        #

        obj_val = frame.FindVariable("cpp_scalar")
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation
        value = obj_val.EvaluateExpression("1")
        self.assertFalse(value.IsValid())
        self.assertFalse(value.GetError().Success())

        #
        # Test C++ array
        #

        obj_val = frame.FindVariable("cpp_array")
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation
        value = obj_val.EvaluateExpression("1")
        self.assertFalse(value.IsValid())
        self.assertFalse(value.GetError().Success())

        # Test retrieveing of an element's field
        value = obj_val.GetValueForExpressionPath("[7]").EvaluateExpression("field")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 1111)

        #
        # Test C++ pointer
        #

        obj_val = frame.FindVariable("cpp_pointer")
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation
        value = obj_val.EvaluateExpression("1")
        self.assertFalse(value.IsValid())
        self.assertFalse(value.GetError().Success())

        # Test retrieveing of a dereferenced object's field
        value = obj_val.Dereference().EvaluateExpression("field")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 1111)

        #
        # Test C++ computation result
        #

        obj_val = frame.EvaluateExpression("cpp_namespace::GetCppStruct()")
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation
        value = obj_val.EvaluateExpression("1")
        self.assertTrue(value.IsValid())
        self.assertFalse(value.GetError().Success())

        #
        # Test C++ computation result located in debuggee memory
        #

        obj_val = frame.EvaluateExpression("cpp_namespace::GetCppStructPtr()")
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation
        value = obj_val.EvaluateExpression("1")
        self.assertFalse(value.IsValid())
        self.assertFalse(value.GetError().Success())

        # Test retrieveing of a dereferenced object's field
        value = obj_val.Dereference().EvaluateExpression("field")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 1111)

    def setUp(self):
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
