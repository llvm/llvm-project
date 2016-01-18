//===-- PythonDataObjectsTests.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#include "Plugins/ScriptInterpreter/Python/PythonDataObjects.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"

using namespace lldb_private;

class PythonDataObjectsTest : public testing::Test
{
  public:
    void
    SetUp() override
    {
        HostInfoBase::Initialize();
        // ScriptInterpreterPython::Initialize() depends on HostInfo being
        // initializedso it can compute the python directory etc.
        ScriptInterpreterPython::Initialize();

        // Although we don't care about concurrency for the purposes of running
        // this test suite, Python requires the GIL to be locked even for
        // deallocating memory, which can happen when you call Py_DECREF or
        // Py_INCREF.  So acquire the GIL for the entire duration of this
        // test suite.
        m_gil_state = PyGILState_Ensure();
    }

    void
    TearDown() override
    {
        PyGILState_Release(m_gil_state);

        ScriptInterpreterPython::Terminate();
    }

  private:
    PyGILState_STATE m_gil_state;
};

TEST_F(PythonDataObjectsTest, TestOwnedReferences)
{
    // After creating a new object, the refcount should be >= 1
    PyObject *obj = PyLong_FromLong(3);
    Py_ssize_t original_refcnt = obj->ob_refcnt;
    EXPECT_LE(1, original_refcnt);

    // If we take an owned reference, the refcount should be the same
    PythonObject owned_long(PyRefType::Owned, obj);
    EXPECT_EQ(original_refcnt, owned_long.get()->ob_refcnt);

    // Take another reference and verify that the refcount increases by 1
    PythonObject strong_ref(owned_long);
    EXPECT_EQ(original_refcnt + 1, strong_ref.get()->ob_refcnt);

    // If we reset the first one, the refcount should be the original value.
    owned_long.Reset();
    EXPECT_EQ(original_refcnt, strong_ref.get()->ob_refcnt);
}

TEST_F(PythonDataObjectsTest, TestResetting)
{
    PythonDictionary dict(PyInitialValue::Empty);

    PyObject *new_dict = PyDict_New();
    dict.Reset(PyRefType::Owned, new_dict);
    EXPECT_EQ(new_dict, dict.get());

    dict.Reset(PyRefType::Owned, nullptr);
    EXPECT_EQ(nullptr, dict.get());

    dict.Reset(PyRefType::Owned, PyDict_New());
    EXPECT_NE(nullptr, dict.get());
    dict.Reset();
    EXPECT_EQ(nullptr, dict.get());
}

TEST_F(PythonDataObjectsTest, TestBorrowedReferences)
{
    PythonInteger long_value(PyRefType::Owned, PyLong_FromLong(3));
    Py_ssize_t original_refcnt = long_value.get()->ob_refcnt;
    EXPECT_LE(1, original_refcnt);

    PythonInteger borrowed_long(PyRefType::Borrowed, long_value.get());
    EXPECT_EQ(original_refcnt + 1, borrowed_long.get()->ob_refcnt);
}

TEST_F(PythonDataObjectsTest, TestPythonInteger)
{
// Test that integers behave correctly when wrapped by a PythonInteger.

#if PY_MAJOR_VERSION < 3
    // Verify that `PythonInt` works correctly when given a PyInt object.
    // Note that PyInt doesn't exist in Python 3.x, so this is only for 2.x
    PyObject *py_int = PyInt_FromLong(12);
    EXPECT_TRUE(PythonInteger::Check(py_int));
    PythonInteger python_int(PyRefType::Owned, py_int);

    EXPECT_EQ(PyObjectType::Integer, python_int.GetObjectType());
    EXPECT_EQ(12, python_int.GetInteger());
#endif

    // Verify that `PythonInteger` works correctly when given a PyLong object.
    PyObject *py_long = PyLong_FromLong(12);
    EXPECT_TRUE(PythonInteger::Check(py_long));
    PythonInteger python_long(PyRefType::Owned, py_long);
    EXPECT_EQ(PyObjectType::Integer, python_long.GetObjectType());

    // Verify that you can reset the value and that it is reflected properly.
    python_long.SetInteger(40);
    EXPECT_EQ(40, python_long.GetInteger());

    // Test that creating a `PythonInteger` object works correctly with the
    // int constructor.
    PythonInteger constructed_int(7);
    EXPECT_EQ(7, constructed_int.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestPythonString)
{
    // Test that strings behave correctly when wrapped by a PythonString.

    static const char *test_string = "PythonDataObjectsTest::TestPythonString1";
    static const char *test_string2 = "PythonDataObjectsTest::TestPythonString2";
    static const char *test_string3 = "PythonDataObjectsTest::TestPythonString3";

#if PY_MAJOR_VERSION < 3
    // Verify that `PythonString` works correctly when given a PyString object.
    // Note that PyString doesn't exist in Python 3.x, so this is only for 2.x
    PyObject *py_string = PyString_FromString(test_string);
    EXPECT_TRUE(PythonString::Check(py_string));
    PythonString python_string(PyRefType::Owned, py_string);

    EXPECT_EQ(PyObjectType::String, python_string.GetObjectType());
    EXPECT_STREQ(test_string, python_string.GetString().data());
#else
    // Verify that `PythonString` works correctly when given a PyUnicode object.
    PyObject *py_unicode = PyUnicode_FromString(test_string);
    EXPECT_TRUE(PythonString::Check(py_unicode));
    PythonString python_unicode(PyRefType::Owned, py_unicode);
    EXPECT_EQ(PyObjectType::String, python_unicode.GetObjectType());
    EXPECT_STREQ(test_string, python_unicode.GetString().data());
#endif

    // Test that creating a `PythonString` object works correctly with the
    // string constructor
    PythonString constructed_string(test_string3);
    EXPECT_STREQ(test_string3, constructed_string.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonStringToStr)
{
    const char *c_str = "PythonDataObjectsTest::TestPythonStringToStr";

    PythonString str(c_str);
    EXPECT_STREQ(c_str, str.GetString().str().c_str());

    PythonString str_str = str.Str();
    EXPECT_STREQ(c_str, str_str.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStr)
{
}

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStructuredInteger)
{
    PythonInteger integer(7);
    auto int_sp = integer.CreateStructuredInteger();
    EXPECT_EQ(7, int_sp->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonStringToStructuredString)
{
    static const char *test_string = "PythonDataObjectsTest::TestPythonStringToStructuredString";
    PythonString constructed_string(test_string);
    auto string_sp = constructed_string.CreateStructuredString();
    EXPECT_STREQ(test_string, string_sp->GetStringValue().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonListValueEquality)
{
    // Test that a list which is built through the native
    // Python API behaves correctly when wrapped by a PythonList.
    static const int list_size = 2;
    static const long long_value0 = 5;
    static const char *const string_value1 = "String Index 1";

    PyObject *py_list = PyList_New(2);
    EXPECT_TRUE(PythonList::Check(py_list));
    PythonList list(PyRefType::Owned, py_list);

    PythonObject list_items[list_size];
    list_items[0].Reset(PythonInteger(long_value0));
    list_items[1].Reset(PythonString(string_value1));

    for (int i = 0; i < list_size; ++i)
        list.SetItemAtIndex(i, list_items[i]);

    EXPECT_EQ(list_size, list.GetSize());
    EXPECT_EQ(PyObjectType::List, list.GetObjectType());

    // Verify that the values match
    PythonObject chk_value1 = list.GetItemAtIndex(0);
    PythonObject chk_value2 = list.GetItemAtIndex(1);
    EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
    EXPECT_TRUE(PythonString::Check(chk_value2.get()));

    PythonInteger chk_int(PyRefType::Borrowed, chk_value1.get());
    PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

    EXPECT_EQ(long_value0, chk_int.GetInteger());
    EXPECT_STREQ(string_value1, chk_str.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonListManipulation)
{
    // Test that manipulation of a PythonList behaves correctly when
    // wrapped by a PythonDictionary.

    static const long long_value0 = 5;
    static const char *const string_value1 = "String Index 1";

    PythonList list(PyInitialValue::Empty);
    PythonInteger integer(long_value0);
    PythonString string(string_value1);

    list.AppendItem(integer);
    list.AppendItem(string);
    EXPECT_EQ(2, list.GetSize());

    // Verify that the values match
    PythonObject chk_value1 = list.GetItemAtIndex(0);
    PythonObject chk_value2 = list.GetItemAtIndex(1);
    EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
    EXPECT_TRUE(PythonString::Check(chk_value2.get()));

    PythonInteger chk_int(PyRefType::Borrowed, chk_value1.get());
    PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

    EXPECT_EQ(long_value0, chk_int.GetInteger());
    EXPECT_STREQ(string_value1, chk_str.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonListToStructuredList)
{
    static const long long_value0 = 5;
    static const char *const string_value1 = "String Index 1";

    PythonList list(PyInitialValue::Empty);
    list.AppendItem(PythonInteger(long_value0));
    list.AppendItem(PythonString(string_value1));

    auto array_sp = list.CreateStructuredArray();
    EXPECT_EQ(StructuredData::Type::eTypeInteger, array_sp->GetItemAtIndex(0)->GetType());
    EXPECT_EQ(StructuredData::Type::eTypeString, array_sp->GetItemAtIndex(1)->GetType());

    auto int_sp = array_sp->GetItemAtIndex(0)->GetAsInteger();
    auto string_sp = array_sp->GetItemAtIndex(1)->GetAsString();

    EXPECT_EQ(long_value0, int_sp->GetValue());
    EXPECT_STREQ(string_value1, string_sp->GetValue().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryValueEquality)
{
    // Test that a dictionary which is built through the native
    // Python API behaves correctly when wrapped by a PythonDictionary.
    static const int dict_entries = 2;
    const char *key_0 = "Key 0";
    int key_1 = 1;
    const int value_0 = 0;
    const char *value_1 = "Value 1";

    PythonObject py_keys[dict_entries];
    PythonObject py_values[dict_entries];

    py_keys[0].Reset(PythonString(key_0));
    py_keys[1].Reset(PythonInteger(key_1));
    py_values[0].Reset(PythonInteger(value_0));
    py_values[1].Reset(PythonString(value_1));

    PyObject *py_dict = PyDict_New();
    EXPECT_TRUE(PythonDictionary::Check(py_dict));
    PythonDictionary dict(PyRefType::Owned, py_dict);

    for (int i = 0; i < dict_entries; ++i)
        PyDict_SetItem(py_dict, py_keys[i].get(), py_values[i].get());
    EXPECT_EQ(dict.GetSize(), dict_entries);
    EXPECT_EQ(PyObjectType::Dictionary, dict.GetObjectType());

    // Verify that the values match
    PythonObject chk_value1 = dict.GetItemForKey(py_keys[0]);
    PythonObject chk_value2 = dict.GetItemForKey(py_keys[1]);
    EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
    EXPECT_TRUE(PythonString::Check(chk_value2.get()));

    PythonInteger chk_int(PyRefType::Borrowed, chk_value1.get());
    PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

    EXPECT_EQ(value_0, chk_int.GetInteger());
    EXPECT_STREQ(value_1, chk_str.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryManipulation)
{
    // Test that manipulation of a dictionary behaves correctly when wrapped
    // by a PythonDictionary.
    static const int dict_entries = 2;

    const char *const key_0 = "Key 0";
    const char *const key_1 = "Key 1";
    const long value_0 = 1;
    const char *const value_1 = "Value 1";

    PythonString keys[dict_entries];
    PythonObject values[dict_entries];

    keys[0].Reset(PythonString(key_0));
    keys[1].Reset(PythonString(key_1));
    values[0].Reset(PythonInteger(value_0));
    values[1].Reset(PythonString(value_1));

    PythonDictionary dict(PyInitialValue::Empty);
    for (int i = 0; i < 2; ++i)
        dict.SetItemForKey(keys[i], values[i]);

    EXPECT_EQ(dict_entries, dict.GetSize());

    // Verify that the keys and values match
    PythonObject chk_value1 = dict.GetItemForKey(keys[0]);
    PythonObject chk_value2 = dict.GetItemForKey(keys[1]);
    EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
    EXPECT_TRUE(PythonString::Check(chk_value2.get()));

    PythonInteger chk_int(PyRefType::Borrowed, chk_value1.get());
    PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

    EXPECT_EQ(value_0, chk_int.GetInteger());
    EXPECT_STREQ(value_1, chk_str.GetString().str().c_str());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryToStructuredDictionary)
{
    static const char *const string_key0 = "String Key 0";
    static const char *const string_key1 = "String Key 1";

    static const char *const string_value0 = "String Value 0";
    static const long int_value1 = 7;

    PythonDictionary dict(PyInitialValue::Empty);
    dict.SetItemForKey(PythonString(string_key0), PythonString(string_value0));
    dict.SetItemForKey(PythonString(string_key1), PythonInteger(int_value1));

    auto dict_sp = dict.CreateStructuredDictionary();
    EXPECT_EQ(2, dict_sp->GetSize());

    EXPECT_TRUE(dict_sp->HasKey(string_key0));
    EXPECT_TRUE(dict_sp->HasKey(string_key1));

    auto string_sp = dict_sp->GetValueForKey(string_key0)->GetAsString();
    auto int_sp = dict_sp->GetValueForKey(string_key1)->GetAsInteger();

    EXPECT_STREQ(string_value0, string_sp->GetValue().c_str());
    EXPECT_EQ(int_value1, int_sp->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonFile)
{
    File file(FileSystem::DEV_NULL, File::eOpenOptionRead);
    PythonFile py_file(file, "r");
    EXPECT_TRUE(PythonFile::Check(py_file.get()));
}

TEST_F(PythonDataObjectsTest, TestObjectAttributes)
{
    PythonInteger py_int(42);
    EXPECT_TRUE(py_int.HasAttribute("numerator"));
    EXPECT_FALSE(py_int.HasAttribute("this_should_not_exist"));

    PythonInteger numerator_attr = py_int.GetAttributeValue("numerator").AsType<PythonInteger>();
    EXPECT_TRUE(numerator_attr.IsAllocated());
    EXPECT_EQ(42, numerator_attr.GetInteger());
}