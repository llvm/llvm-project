"""
Test that we are able to properly report a usable dynamic type
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCDynamicSBTypeTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    @skipIf(archs="i[3-6]86")
    def test_dyn(self):
        """Test that we are able to properly report a usable dynamic type."""
        d = {'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            self.main_source,
            self.line,
            num_expected_locations=1,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        v_object = self.frame().FindVariable(
            "object").GetDynamicValue(lldb.eDynamicCanRunTarget)
        v_base = self.frame().FindVariable(
            "base").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertEqual(
            v_object.GetTypeName(), "MyDerivedClass *",
            "The NSObject is properly type-named")
        self.assertEqual(
            v_base.GetTypeName(), "MyDerivedClass *",
            "The Base is properly type-named")
        object_type = v_object.GetType()
        base_type = v_base.GetType()
        self.assertEqual(
            object_type.GetName(), "MyDerivedClass *",
            "The dynamic SBType for NSObject is for the correct type")
        self.assertEqual(
            base_type.GetName(), "MyDerivedClass *",
            "The dynamic SBType for Base is for the correct type")
        object_pointee_type = object_type.GetPointeeType()
        base_pointee_type = base_type.GetPointeeType()
        self.assertEqual(
            object_pointee_type.GetName(), "MyDerivedClass",
            "The dynamic type for NSObject figures out its pointee type just fine")
        self.assertEqual(
            base_pointee_type.GetName(), "MyDerivedClass",
            "The dynamic type for Base figures out its pointee type just fine")

        self.assertEqual(
            object_pointee_type.GetDirectBaseClassAtIndex(0).GetName(), "MyBaseClass",
            "The dynamic type for NSObject can go back to its base class")
        self.assertEqual(
            base_pointee_type.GetDirectBaseClassAtIndex(0).GetName(), "MyBaseClass",
            "The dynamic type for Base can go back to its base class")

        self.assertEqual(
            object_pointee_type.GetDirectBaseClassAtIndex(0).GetType().GetDirectBaseClassAtIndex(0).GetName(),
            "NSObject", "The dynamic type for NSObject can go up the hierarchy")
        self.assertEqual(
            base_pointee_type.GetDirectBaseClassAtIndex(0).GetType().GetDirectBaseClassAtIndex(0).GetName(),
            "NSObject", "The dynamic type for Base can go up the hierarchy")

        self.assertEqual(
            object_pointee_type.GetNumberOfFields(), 2,
            "The dynamic type for NSObject has 2 fields")
        self.assertEqual(
            base_pointee_type.GetNumberOfFields(), 2,
            "The dynamic type for Base has 2 fields")
