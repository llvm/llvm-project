"""
Test some SBStructuredData API.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStructuredDataAPI(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.structured_data_api_test()

    @add_test_categories(['pyapi'])
    def structured_data_api_test(self):
        error = lldb.SBError()
        s = lldb.SBStream()
        s.Print(
            "{\"key_dict\":{\"key_string\":\"STRING\",\"key_int\":3,\"key_float\":2.99,\"key_bool\":true,\"key_array\":[\"23\",\"arr\"]}}")
        example = lldb.SBStructuredData()

        # Check SetFromJSON API for dictionaries, integers, floating point
        # values, strings and arrays
        error = example.SetFromJSON(s)
        if not error.Success():
            self.fail("FAILED:   " + error.GetCString())

        # Tests for invalid data type
        self.invalid_struct_test(example)

        dict_struct = lldb.SBStructuredData()
        dict_struct = example.GetValueForKey("key_dict")

        # Tests for dictionary data type
        self.dictionary_struct_test(example)

        # Tests for string data type
        self.string_struct_test(dict_struct)

        # Tests for integer data type
        self.int_struct_test(dict_struct)

        # Tests for floating point data type
        self.double_struct_test(dict_struct)

        # Tests for boolean data type
        self.bool_struct_test(dict_struct)

        # Tests for array data type
        self.array_struct_test(dict_struct)

    def invalid_struct_test(self, example):
        invalid_struct = lldb.SBStructuredData()
        invalid_struct = example.GetValueForKey("invalid_key")
        if invalid_struct.IsValid():
            self.fail("An invalid object should have been returned")

        # Check Type API
        if not invalid_struct.GetType() == lldb.eStructuredDataTypeInvalid:
            self.fail("Wrong type returned: " + str(invalid_struct.GetType()))

    def dictionary_struct_test(self, example):
        # Check API returning a valid SBStructuredData of 'dictionary' type
        dict_struct = lldb.SBStructuredData()
        dict_struct = example.GetValueForKey("key_dict")
        if not dict_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not dict_struct.GetType() == lldb.eStructuredDataTypeDictionary:
            self.fail("Wrong type returned: " + str(dict_struct.GetType()))

        # Check Size API for 'dictionary' type
        if not dict_struct.GetSize() == 5:
            self.fail("Wrong no of elements returned: " +
                      str(dict_struct.GetSize()))

    def string_struct_test(self, dict_struct):
        string_struct = lldb.SBStructuredData()
        string_struct = dict_struct.GetValueForKey("key_string")
        if not string_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not string_struct.GetType() == lldb.eStructuredDataTypeString:
            self.fail("Wrong type returned: " + str(string_struct.GetType()))

        # Check API returning 'string' value
        output = string_struct.GetStringValue(25)
        if not "STRING" in output:
            self.fail("wrong output: " + output)

        # Calling wrong API on a SBStructuredData
        # (e.g. getting an integer from a string type structure)
        output = string_struct.GetIntegerValue()
        if output:
            self.fail(
                "Valid integer value " +
                str(output) +
                " returned for a string object")

    def int_struct_test(self, dict_struct):
        # Check a valid SBStructuredData containing an 'integer' by
        int_struct = lldb.SBStructuredData()
        int_struct = dict_struct.GetValueForKey("key_int")
        if not int_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not int_struct.GetType() == lldb.eStructuredDataTypeInteger:
            self.fail("Wrong type returned: " + str(int_struct.GetType()))

        # Check API returning 'integer' value
        output = int_struct.GetIntegerValue()
        if not output == 3:
            self.fail("wrong output: " + str(output))

        # Calling wrong API on a SBStructuredData
        # (e.g. getting a string value from an integer type structure)
        output = int_struct.GetStringValue(25)
        if output:
            self.fail(
                "Valid string " +
                output +
                " returned for an integer object")

    def double_struct_test(self, dict_struct):
        floating_point_struct = lldb.SBStructuredData()
        floating_point_struct = dict_struct.GetValueForKey("key_float")
        if not floating_point_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not floating_point_struct.GetType() == lldb.eStructuredDataTypeFloat:
            self.fail("Wrong type returned: " +
                      str(floating_point_struct.GetType()))

        # Check API returning 'double' value
        output = floating_point_struct.GetFloatValue()
        if not output == 2.99:
            self.fail("wrong output: " + str(output))

    def bool_struct_test(self, dict_struct):
        bool_struct = lldb.SBStructuredData()
        bool_struct = dict_struct.GetValueForKey("key_bool")
        if not bool_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not bool_struct.GetType() == lldb.eStructuredDataTypeBoolean:
            self.fail("Wrong type returned: " + str(bool_struct.GetType()))

        # Check API returning 'bool' value
        output = bool_struct.GetBooleanValue()
        if not output:
            self.fail("wrong output: " + str(output))

    def array_struct_test(self, dict_struct):
        # Check API returning a valid SBStructuredData of 'array' type
        array_struct = lldb.SBStructuredData()
        array_struct = dict_struct.GetValueForKey("key_array")
        if not array_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not array_struct.GetType() == lldb.eStructuredDataTypeArray:
            self.fail("Wrong type returned: " + str(array_struct.GetType()))

        # Check Size API for 'array' type
        if not array_struct.GetSize() == 2:
            self.fail("Wrong no of elements returned: " +
                      str(array_struct.GetSize()))

        # Check API returning a valid SBStructuredData for different 'array'
        # indices
        string_struct = array_struct.GetItemAtIndex(0)
        if not string_struct.IsValid():
            self.fail("A valid object should have been returned")
        if not string_struct.GetType() == lldb.eStructuredDataTypeString:
            self.fail("Wrong type returned: " + str(string_struct.GetType()))
        output = string_struct.GetStringValue(5)
        if not output == "23":
            self.fail("wrong output: " + str(output))

        string_struct = array_struct.GetItemAtIndex(1)
        if not string_struct.IsValid():
            self.fail("A valid object should have been returned")
        if not string_struct.GetType() == lldb.eStructuredDataTypeString:
            self.fail("Wrong type returned: " + str(string_struct.GetType()))
        output = string_struct.GetStringValue(5)
        if not output == "arr":
            self.fail("wrong output: " + str(output))
