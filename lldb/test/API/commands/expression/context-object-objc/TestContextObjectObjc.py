"""
Tests expression evaluation in context of an objc class.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class ContextObjectObjcTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_context_object_objc(self):
        """Tests expression evaluation in context of an objc class."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, '// Break here', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        #
        # Test objc class variable
        #

        obj_val = frame.FindVariable("objcClass")
        self.assertTrue(obj_val.IsValid())
        obj_val = obj_val.Dereference()
        self.assertTrue(obj_val.IsValid())

        # Test an empty expression evaluation
        value = obj_val.EvaluateExpression("")
        self.assertFalse(value.IsValid())
        self.assertFalse(value.GetError().Success())

        # Test retrieving of a field (not a local with the same name)
        value = obj_val.EvaluateExpression("field")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(), 1111)

        # Test if the self pointer is properly evaluated

        # Test retrieving of an objcClass's property through the self pointer
        value = obj_val.EvaluateExpression("self.property")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 2222)

        # Test objcClass's methods evaluation through the self pointer
        value = obj_val.EvaluateExpression("[self method]")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 3333)

        # Test if we can use a computation result reference object correctly

        obj_val = frame.EvaluateExpression("[ObjcClass createNew]")
        self.assertTrue(obj_val.IsValid())
        obj_val = obj_val.Dereference()
        self.assertTrue(obj_val.IsValid())

        # Test an expression evaluation on it
        value = obj_val.EvaluateExpression("1")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())

        # Test retrieving of a field on it
        value = obj_val.EvaluateExpression("field")
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsSigned(), 1111)

    def setUp(self):
        TestBase.setUp(self)

        self.main_source = "main.m"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
