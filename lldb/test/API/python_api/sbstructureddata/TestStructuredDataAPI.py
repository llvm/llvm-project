"""
Test some SBStructuredData API.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json


class TestStructuredDataAPI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.structured_data_api_test()

    def structured_data_api_test(self):
        error = lldb.SBError()
        s = lldb.SBStream()

        dict_str = json.dumps(
            {
                "key_dict": {
                    "key_string": "STRING",
                    "key_uint": 0xFFFFFFFF00000000,
                    "key_sint": -42,
                    "key_float": 2.99,
                    "key_bool": True,
                    "key_array": ["23", "arr"],
                }
            }
        )
        s.Print(dict_str)
        example = lldb.SBStructuredData()

        # Check SetFromJSON API for dictionaries, integers, floating point
        # values, strings and arrays
        error = example.SetFromJSON(s)
        if not error.Success():
            self.fail("FAILED:   " + error.GetCString())

        # Tests for invalid data type
        self.invalid_struct_test(example)

        # Test that GetDescription works:
        s.Clear()
        error = example.GetDescription(s)
        self.assertSuccess(error, "GetDescription works")
        if not "key_float" in s.GetData():
            self.fail("FAILED: could not find key_float in description output")

        dict_struct = lldb.SBStructuredData()
        dict_struct = example.GetValueForKey("key_dict")

        # Tests for dictionary data type
        self.dictionary_struct_test(example)

        # Tests for string data type
        self.string_struct_test(dict_struct)

        # Tests for integer data type
        self.uint_struct_test(dict_struct)

        # Tests for integer data type
        self.sint_struct_test(dict_struct)

        # Tests for floating point data type
        self.double_struct_test(dict_struct)

        # Tests for boolean data type
        self.bool_struct_test(dict_struct)

        # Tests for array data type
        self.array_struct_test(dict_struct)

        s.Clear()
        self.assertSuccess(example.GetAsJSON(s))
        py_obj = json.loads(s.GetData())
        self.assertTrue(py_obj)
        self.assertIn("key_dict", py_obj)

        py_dict = py_obj["key_dict"]
        self.assertEqual(py_dict["key_string"], "STRING")
        self.assertEqual(py_dict["key_uint"], 0xFFFFFFFF00000000)
        self.assertEqual(py_dict["key_sint"], -42)
        self.assertEqual(py_dict["key_float"], 2.99)
        self.assertEqual(py_dict["key_bool"], True)
        self.assertEqual(py_dict["key_array"], ["23", "arr"])

        class MyRandomClass:
            payload = "foo"

        py_dict["key_generic"] = MyRandomClass()

        stp = lldb.SBScriptObject(py_dict, lldb.eScriptLanguagePython)
        self.assertEqual(stp.ptr, py_dict)

        sd = lldb.SBStructuredData(stp, self.dbg)
        self.assertTrue(sd.IsValid())
        self.assertEqual(sd.GetSize(), len(py_dict))

        generic_sd = sd.GetValueForKey("key_generic")
        self.assertTrue(generic_sd.IsValid())
        self.assertEqual(generic_sd.GetType(), lldb.eStructuredDataTypeGeneric)

        my_random_class = generic_sd.GetGenericValue()
        self.assertTrue(my_random_class)
        self.assertEqual(my_random_class.payload, MyRandomClass.payload)

        example = lldb.SBStructuredData()
        self.assertSuccess(example.SetFromJSON("1"))
        self.assertEqual(example.GetType(), lldb.eStructuredDataTypeInteger)
        self.assertEqual(example.GetIntegerValue(), 1)

        self.assertSuccess(example.SetFromJSON("4.19"))
        self.assertEqual(example.GetType(), lldb.eStructuredDataTypeFloat)
        self.assertEqual(example.GetFloatValue(), 4.19)

        self.assertSuccess(example.SetFromJSON('"Bonjour, 123!"'))
        self.assertEqual(example.GetType(), lldb.eStructuredDataTypeString)
        self.assertEqual(example.GetStringValue(42), "Bonjour, 123!")

        self.assertSuccess(example.SetFromJSON("true"))
        self.assertEqual(example.GetType(), lldb.eStructuredDataTypeBoolean)
        self.assertTrue(example.GetBooleanValue())

        self.assertSuccess(example.SetFromJSON("null"))
        self.assertEqual(example.GetType(), lldb.eStructuredDataTypeNull)

        example_arr = [1, 2.3, "4", {"5": False}]
        arr_str = json.dumps(example_arr)
        s.Clear()
        s.Print(arr_str)
        self.assertSuccess(example.SetFromJSON(s))

        s.Clear()
        self.assertSuccess(example.GetAsJSON(s))
        sb_data = json.loads(s.GetData())
        self.assertEqual(sb_data, example_arr)

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
        if not dict_struct.GetSize() == 6:
            self.fail("Wrong no of elements returned: " + str(dict_struct.GetSize()))

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
                "Valid integer value " + str(output) + " returned for a string object"
            )

    def uint_struct_test(self, dict_struct):
        # Check a valid SBStructuredData containing an unsigned integer.
        # We intentionally make this larger than what an int64_t can hold but
        # still small enough to fit a uint64_t
        uint_struct = lldb.SBStructuredData()
        uint_struct = dict_struct.GetValueForKey("key_uint")
        if not uint_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not uint_struct.GetType() == lldb.eStructuredDataTypeInteger:
            self.fail("Wrong type returned: " + str(uint_struct.GetType()))

        # Check API returning unsigned integer value
        output = uint_struct.GetUnsignedIntegerValue()
        if not output == 0xFFFFFFFF00000000:
            self.fail("wrong output: " + str(output))

        # Calling wrong API on a SBStructuredData
        # (e.g. getting a string value from an integer type structure)
        output = uint_struct.GetStringValue(25)
        if output:
            self.fail("Valid string " + output + " returned for an integer object")

    def sint_struct_test(self, dict_struct):
        # Check a valid SBStructuredData containing an signed integer.
        # We intentionally make this smaller than what an uint64_t can hold but
        # still small enough to fit a int64_t
        sint_struct = lldb.SBStructuredData()
        sint_struct = dict_struct.GetValueForKey("key_sint")
        if not sint_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not sint_struct.GetType() == lldb.eStructuredDataTypeSignedInteger:
            self.fail("Wrong type returned: " + str(sint_struct.GetType()))

        # Check API returning signed integer value
        output = sint_struct.GetSignedIntegerValue()
        if not output == -42:
            self.fail("wrong output: " + str(output))

        # Calling wrong API on a SBStructuredData
        # (e.g. getting a string value from an integer type structure)
        output = sint_struct.GetStringValue(69)
        if output:
            self.fail("Valid string " + output + " returned for an integer object")

    def double_struct_test(self, dict_struct):
        floating_point_struct = lldb.SBStructuredData()
        floating_point_struct = dict_struct.GetValueForKey("key_float")
        if not floating_point_struct.IsValid():
            self.fail("A valid object should have been returned")

        # Check Type API
        if not floating_point_struct.GetType() == lldb.eStructuredDataTypeFloat:
            self.fail("Wrong type returned: " + str(floating_point_struct.GetType()))

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
            self.fail("Wrong no of elements returned: " + str(array_struct.GetSize()))

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
