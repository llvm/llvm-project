"""
Make sure FindTypes finds struct types with the struct prefix.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFindTypesOnStructType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_find_types_struct_type(self):
        """Make sure FindTypes actually finds 'struct typename' not just 'typename'."""
        self.build()
        self.do_test()

    def do_test(self):
        """Make sure FindTypes actually finds 'struct typename' not just 'typename'."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Make sure this works with struct
        type_list = target.FindTypes("struct mytype")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type with struct")

        # Make sure this works without the struct:
        type_list = target.FindTypes("mytype")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type without struct")

        # Make sure it works with union
        type_list = target.FindTypes("union myunion")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type with union")

        # Make sure this works without the union:
        type_list = target.FindTypes("myunion")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type without union")

        # Make sure it works with typedef
        type_list = target.FindTypes("typedef MyType")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type with typedef")

        # Make sure this works without the typedef:
        type_list = target.FindTypes("MyType")
        self.assertEqual(type_list.GetSize(), 1, "Found one instance of the type without typedef")



