"""
Test the SBModule and SBTarget type lookup APIs to find multiple types.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeFindFirstTestCase(TestBase):
    def test_find_first_type(self):
        """
        Test SBTarget::FindTypes() and SBModule::FindTypes() APIs.

        We had issues where our declaration context when finding types was
        incorrectly calculated where a type in a namepace, and a type in a
        function that was also in the same namespace would match a lookup. For
        example:

            namespace a {
              struct Foo {
                int foo;
              };

              unsigned foo() {
                typedef unsigned Foo;
                Foo foo = 12;
                return foo;
              }
            } // namespace a


        Previously LLDB would calculate the declaration context of "a::Foo"
        correctly, but incorrectly calculate the declaration context of "Foo"
        from within the foo() function as "a::Foo". Adding tests to ensure this
        works correctly.
        """
        self.build()
        target = self.createTestTarget()
        exe_module = target.GetModuleAtIndex(0)
        self.assertTrue(exe_module.IsValid())
        # Test the SBTarget and SBModule APIs for FindFirstType
        for api in [target, exe_module]:
            # We should find the "a::Foo" but not the "Foo" type in the function
            types = api.FindTypes("a::Foo")
            self.assertEqual(types.GetSize(), 1)
            type_str0 = str(types.GetTypeAtIndex(0))
            self.assertIn('struct Foo {', type_str0)

            # When we search by type basename, we should find any type whose
            # basename matches "Foo", so "a::Foo" and the "Foo" type in the
            # function.
            types = api.FindTypes("Foo")
            self.assertEqual(types.GetSize(), 2)
            type_str0 = str(types.GetTypeAtIndex(0))
            type_str1 = str(types.GetTypeAtIndex(1))
            # We don't know which order the types will come back as, so
            self.assertEqual(set([str(t).split('\n')[0] for t in types]), set(["typedef Foo", "struct Foo {"]))

            # When we search by type basename with "::" prepended, we should
            # only types in the root namespace which means only "Foo" type in
            # the function.
            types = api.FindTypes("::Foo")
            self.assertEqual(types.GetSize(), 1)
            type_str0 = str(types.GetTypeAtIndex(0))
            self.assertIn('typedef Foo', type_str0)
