"""
Test SBType for ObjC classes.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCSBTypeTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line = line_number("main.m", "// Break at this line")

    @add_test_categories(["objc", "pyapi"])
    def test(self):
        """Test SBType for ObjC classes."""
        self.build()
        lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec("main.m"), self.line)

        aBar = self.frame().FindVariable("aBar")
        aBarType = aBar.GetType()
        self.assertTrue(aBarType.IsValid(), "Bar should be a valid data type")
        self.assertEqual(aBarType.GetName(), "Bar *", "Bar has the right name")

        self.assertEqual(
            aBarType.GetNumberOfDirectBaseClasses(), 1, "Bar has a superclass"
        )
        aFooType = aBarType.GetDirectBaseClassAtIndex(0)

        self.assertTrue(aFooType.IsValid(), "Foo should be a valid data type")
        self.assertEqual(aFooType.GetName(), "Foo", "Foo has the right name")

        self.assertEqual(aBarType.GetNumberOfFields(), 1, "Bar has a field")
        aBarField = aBarType.GetFieldAtIndex(0)

        self.assertEqual(aBarField.GetName(), "_iVar", "The field has the right name")
