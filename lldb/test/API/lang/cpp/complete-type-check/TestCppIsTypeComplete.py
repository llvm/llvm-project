""" Tests SBType.IsTypeComplete on C++ types. """


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def assertComplete(self, typename):
        """ Asserts that the type with the given name is complete. """
        found_type = self.target().FindFirstType(typename)
        self.assertTrue(found_type.IsValid())
        self.assertTrue(found_type.IsTypeComplete())

    def assertIsNotPresent(self, typename):
        """ Asserts that the type with the given name is not found. """
        found_type = self.target().FindFirstType(typename)
        self.assertFalse(found_type.IsValid())

    def assertCompleteWithVar(self, typename):
        """ Asserts that the type with the given name is complete. """
        found_type = self.target().FindFirstType(typename)
        self.assertTrue(found_type.IsValid())
        self.assertTrue(found_type.IsTypeComplete())

    def assertPointeeIncomplete(self, typename, variable):
        """ Asserts that the pointee type behind the type with the given name
        is not complete. The variable is used to find the type."""
        found_type = self.target().FindFirstType(typename)
        found_type = self.expect_expr(variable, result_type=typename).GetType()
        self.assertTrue(found_type.IsPointerType())
        pointee_type = found_type.GetPointeeType()
        self.assertTrue(pointee_type.IsValid())
        self.assertFalse(pointee_type.IsTypeComplete())

    @no_debug_info_test
    def test_forward_declarations(self):
        """ Tests types of declarations that can be forward declared. """
        self.build()
        self.createTestTarget()

        # Check record types with a definition.
        self.assertCompleteWithVar("EmptyClass")
        self.assertCompleteWithVar("DefinedClass")
        self.assertCompleteWithVar("DefinedClassTypedef")
        self.assertCompleteWithVar("DefinedTemplateClass<int>")
        self.assertIsNotPresent("DefinedTemplateClass<long>")

        # Record types without a defining declaration are not complete.
        self.assertPointeeIncomplete("FwdClass *", "fwd_class")
        self.assertPointeeIncomplete("FwdClassTypedef *", "fwd_class_typedef")
        self.assertPointeeIncomplete("FwdTemplateClass<int> *", "fwd_template_class")

        # A pointer type is complete even when it points to an incomplete type.
        fwd_class_ptr = self.expect_expr("fwd_class", result_type="FwdClass *")
        self.assertTrue(fwd_class_ptr.GetType().IsTypeComplete())

    @no_debug_info_test
    def test_builtin_types(self):
        """ Tests builtin types and types derived from them. """
        self.build()
        self.createTestTarget()

        # Void is complete.
        void_type = self.target().FindFirstType("void")
        self.assertTrue(void_type.IsValid())
        self.assertTrue(void_type.IsTypeComplete())

        # Builtin types are also complete.
        int_type = self.target().FindFirstType("int")
        self.assertTrue(int_type.IsValid())
        self.assertTrue(int_type.IsTypeComplete())

        # References to builtin types are also complete.
        int_ref_type = int_type.GetReferenceType()
        self.assertTrue(int_ref_type.IsValid())
        self.assertTrue(int_ref_type.IsTypeComplete())

        # Pointer types to basic types are always complete.
        int_ptr_type = int_type.GetReferenceType()
        self.assertTrue(int_ptr_type.IsValid())
        self.assertTrue(int_ptr_type.IsTypeComplete())
