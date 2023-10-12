"""
Test SBType and SBTypeList API.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeAndTypeListTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break at.
        self.source = "main.cpp"
        self.line = line_number(self.source, "// Break at this line")

    def test(self):
        """Exercise SBType and SBTypeList API."""
        d = {"EXE": self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = self.getBuildArtifact(self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition",
        )
        frame0 = thread.GetFrameAtIndex(0)

        # Get the type 'Task'.
        type_list = target.FindTypes("Task")
        if self.TraceOn():
            print(
                "Size of type_list from target.FindTypes('Task') query: %d"
                % type_list.GetSize()
            )
        # a second Task make be scared up by the Objective-C runtime
        self.assertTrue(len(type_list) >= 1)
        for type in type_list:
            self.assertTrue(type)
            self.DebugSBType(type)
            self.assertFalse(type.IsAnonymousType(), "Task is not anonymous")
            self.assertTrue(type.IsAggregateType(), "Task is aggregate")
            for field in type.fields:
                if field.name == "type":
                    for enum_member in field.type.enum_members:
                        self.assertTrue(enum_member)
                        self.DebugSBType(enum_member.type)
                elif field.name == "my_type_is_nameless":
                    self.assertFalse(
                        field.type.IsAnonymousType(),
                        "my_type_is_nameless is not an anonymous type",
                    )
                    self.assertTrue(field.type.IsAggregateType())
                elif field.name == "my_type_is_named":
                    self.assertFalse(
                        field.type.IsAnonymousType(),
                        "my_type_is_named has a named type",
                    )
                    self.assertTrue(field.type.IsAggregateType())
                elif field.name == None:
                    self.assertTrue(
                        field.type.IsAnonymousType(), "Nameless type is not anonymous"
                    )

        # Pass an empty string.  LLDB should not crash. :-)
        fuzz_types = target.FindTypes(None)
        fuzz_type = target.FindFirstType(None)

        # Now use the SBTarget.FindFirstType() API to find 'Task'.
        task_type = target.FindFirstType("Task")
        self.assertTrue(task_type)
        self.DebugSBType(task_type)

        # Get the reference type of 'Task', just for fun.
        task_ref_type = task_type.GetReferenceType()
        self.assertTrue(task_ref_type)
        self.DebugSBType(task_ref_type)

        # Get the pointer type of 'Task', which is the same as task_head's
        # type.
        task_pointer_type = task_type.GetPointerType()
        self.assertTrue(task_pointer_type)
        self.DebugSBType(task_pointer_type)

        # Get variable 'task_head'.
        task_head = frame0.FindVariable("task_head")
        self.assertTrue(task_head, VALID_VARIABLE)
        self.DebugSBValue(task_head)
        task_head_type = task_head.GetType()
        self.DebugSBType(task_head_type)
        self.assertTrue(task_head_type.IsPointerType())
        self.assertFalse(task_head_type.IsAggregateType())

        self.assertEqual(task_head_type, task_pointer_type)

        # Get the pointee type of 'task_head'.
        task_head_pointee_type = task_head_type.GetPointeeType()
        self.DebugSBType(task_head_pointee_type)

        self.assertEqual(task_type, task_head_pointee_type)

        # Check whether we can find a directly nested type by name
        name_type = task_type.FindDirectNestedType("name")
        self.assertTrue(name_type)
        self.DebugSBType(name_type)

        enum_type = task_type.FindDirectNestedType("E")
        self.assertTrue(enum_type)
        self.DebugSBType(enum_type)

        union_type = task_type.FindDirectNestedType("U")
        self.assertTrue(union_type)
        self.DebugSBType(union_type)

        # Check that we don't find indirectly nested types

        self.assertTrue(enum_type.size == 1)

        invalid_type = task_type.FindDirectNestedType("E2")
        self.assertFalse(invalid_type)

        # Check that FindDirectNestedType handles types without DeclContext
        # and other errorneous inputs
        task_ptr_type = task_type.GetPointerType()
        invalid_type = task_ptr_type.FindDirectNestedType("name")
        self.assertFalse(invalid_type)

        invalid_type = task_type.FindDirectNestedType("")
        self.assertFalse(invalid_type)

        invalid_type = task_type.FindDirectNestedType(None)
        self.assertFalse(invalid_type)

        # We'll now get the child member 'id' from 'task_head'.
        id = task_head.GetChildMemberWithName("id")
        self.DebugSBValue(id)
        id_type = id.GetType()
        self.DebugSBType(id_type)
        self.assertFalse(id_type.IsAggregateType())

        # SBType.GetBasicType() takes an enum 'BasicType'
        # (lldb-enumerations.h).
        int_type = id_type.GetBasicType(lldb.eBasicTypeInt)
        self.assertEqual(id_type, int_type)

        # Find 'myint_arr' and check the array element type.
        myint_arr = frame0.FindVariable("myint_arr")
        self.assertTrue(myint_arr, VALID_VARIABLE)
        self.DebugSBValue(myint_arr)
        myint_arr_type = myint_arr.GetType()
        self.DebugSBType(myint_arr_type)
        self.assertTrue(myint_arr_type.IsArrayType())
        self.assertTrue(myint_arr_type.IsAggregateType())
        myint_arr_element_type = myint_arr_type.GetArrayElementType()
        self.DebugSBType(myint_arr_element_type)
        myint_type = target.FindFirstType("myint")
        self.DebugSBType(myint_type)
        self.assertEqual(myint_arr_element_type, myint_type)

        # Test enum methods. Requires DW_AT_enum_class which was added in Dwarf 4.
        if configuration.dwarf_version >= 4:
            enum_type = target.FindFirstType("EnumType")
            self.assertTrue(enum_type)
            self.DebugSBType(enum_type)
            self.assertFalse(enum_type.IsScopedEnumerationType())
            self.assertFalse(enum_type.IsAggregateType())

            scoped_enum_type = target.FindFirstType("ScopedEnumType")
            self.assertTrue(scoped_enum_type)
            self.DebugSBType(scoped_enum_type)
            self.assertTrue(scoped_enum_type.IsScopedEnumerationType())
            self.assertFalse(scoped_enum_type.IsAggregateType())
            int_scoped_enum_type = scoped_enum_type.GetEnumerationIntegerType()
            self.assertTrue(int_scoped_enum_type)
            self.DebugSBType(int_scoped_enum_type)
            self.assertEquals(int_scoped_enum_type.GetName(), "int")

            enum_uchar = target.FindFirstType("EnumUChar")
            self.assertTrue(enum_uchar)
            self.DebugSBType(enum_uchar)
            int_enum_uchar = enum_uchar.GetEnumerationIntegerType()
            self.assertTrue(int_enum_uchar)
            self.DebugSBType(int_enum_uchar)
            self.assertEquals(int_enum_uchar.GetName(), "unsigned char")
