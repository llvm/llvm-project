//===-- PythonDataObjectsTests.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#include "gtest/gtest.h"

#include "Plugins/ScriptInterpreter/Python/PythonDataObjects.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"

#include "PythonTestSuite.h"

using namespace lldb_private;

class PythonDataObjectsTest : public PythonTestSuite {
public:
  void SetUp() override {
    PythonTestSuite::SetUp();

    PythonString sys_module("sys");
    m_sys_module.Reset(PyRefType::Owned, PyImport_Import(sys_module.get()));
    m_main_module = PythonModule::MainModule();
    m_builtins_module = PythonModule::BuiltinsModule();
  }

  void TearDown() override {
    m_sys_module.Reset();
    m_main_module.Reset();
    m_builtins_module.Reset();

    PythonTestSuite::TearDown();
  }

protected:
  PythonModule m_sys_module;
  PythonModule m_main_module;
  PythonModule m_builtins_module;
};

TEST_F(PythonDataObjectsTest, TestOwnedReferences) {
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

TEST_F(PythonDataObjectsTest, TestResetting) {
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

TEST_F(PythonDataObjectsTest, TestBorrowedReferences) {
  PythonInteger long_value(PyRefType::Owned, PyLong_FromLong(3));
  Py_ssize_t original_refcnt = long_value.get()->ob_refcnt;
  EXPECT_LE(1, original_refcnt);

  PythonInteger borrowed_long(PyRefType::Borrowed, long_value.get());
  EXPECT_EQ(original_refcnt + 1, borrowed_long.get()->ob_refcnt);
}

TEST_F(PythonDataObjectsTest, TestGlobalNameResolutionNoDot) {
  PythonObject sys_module = m_main_module.ResolveName("sys");
  EXPECT_EQ(m_sys_module.get(), sys_module.get());
  EXPECT_TRUE(sys_module.IsAllocated());
  EXPECT_TRUE(PythonModule::Check(sys_module.get()));
}

TEST_F(PythonDataObjectsTest, TestModuleNameResolutionNoDot) {
  PythonObject sys_path = m_sys_module.ResolveName("path");
  PythonObject sys_version_info = m_sys_module.ResolveName("version_info");
  EXPECT_TRUE(sys_path.IsAllocated());
  EXPECT_TRUE(sys_version_info.IsAllocated());

  EXPECT_TRUE(PythonList::Check(sys_path.get()));
}

TEST_F(PythonDataObjectsTest, TestTypeNameResolutionNoDot) {
  PythonObject sys_version_info = m_sys_module.ResolveName("version_info");

  PythonObject version_info_type(PyRefType::Owned,
                                 PyObject_Type(sys_version_info.get()));
  EXPECT_TRUE(version_info_type.IsAllocated());
  PythonObject major_version_field = version_info_type.ResolveName("major");
  EXPECT_TRUE(major_version_field.IsAllocated());
}

TEST_F(PythonDataObjectsTest, TestInstanceNameResolutionNoDot) {
  PythonObject sys_version_info = m_sys_module.ResolveName("version_info");
  PythonObject major_version_field = sys_version_info.ResolveName("major");
  PythonObject minor_version_field = sys_version_info.ResolveName("minor");

  EXPECT_TRUE(major_version_field.IsAllocated());
  EXPECT_TRUE(minor_version_field.IsAllocated());

  PythonInteger major_version_value =
      major_version_field.AsType<PythonInteger>();
  PythonInteger minor_version_value =
      minor_version_field.AsType<PythonInteger>();

  EXPECT_EQ(PY_MAJOR_VERSION, major_version_value.GetInteger());
  EXPECT_EQ(PY_MINOR_VERSION, minor_version_value.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestGlobalNameResolutionWithDot) {
  PythonObject sys_path = m_main_module.ResolveName("sys.path");
  EXPECT_TRUE(sys_path.IsAllocated());
  EXPECT_TRUE(PythonList::Check(sys_path.get()));

  PythonInteger version_major =
      m_main_module.ResolveName("sys.version_info.major")
          .AsType<PythonInteger>();
  PythonInteger version_minor =
      m_main_module.ResolveName("sys.version_info.minor")
          .AsType<PythonInteger>();
  EXPECT_TRUE(version_major.IsAllocated());
  EXPECT_TRUE(version_minor.IsAllocated());
  EXPECT_EQ(PY_MAJOR_VERSION, version_major.GetInteger());
  EXPECT_EQ(PY_MINOR_VERSION, version_minor.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestDictionaryResolutionWithDot) {
  // Make up a custom dictionary with "sys" pointing to the `sys` module.
  PythonDictionary dict(PyInitialValue::Empty);
  dict.SetItemForKey(PythonString("sys"), m_sys_module);

  // Now use that dictionary to resolve `sys.version_info.major`
  PythonInteger version_major =
      PythonObject::ResolveNameWithDictionary("sys.version_info.major", dict)
          .AsType<PythonInteger>();
  PythonInteger version_minor =
      PythonObject::ResolveNameWithDictionary("sys.version_info.minor", dict)
          .AsType<PythonInteger>();
  EXPECT_EQ(PY_MAJOR_VERSION, version_major.GetInteger());
  EXPECT_EQ(PY_MINOR_VERSION, version_minor.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestPythonInteger) {
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

TEST_F(PythonDataObjectsTest, TestPythonBytes) {
  static const char *test_bytes = "PythonDataObjectsTest::TestPythonBytes";
  PyObject *py_bytes = PyBytes_FromString(test_bytes);
  EXPECT_TRUE(PythonBytes::Check(py_bytes));
  PythonBytes python_bytes(PyRefType::Owned, py_bytes);

#if PY_MAJOR_VERSION < 3
  EXPECT_TRUE(PythonString::Check(py_bytes));
  EXPECT_EQ(PyObjectType::String, python_bytes.GetObjectType());
#else
  EXPECT_FALSE(PythonString::Check(py_bytes));
  EXPECT_EQ(PyObjectType::Bytes, python_bytes.GetObjectType());
#endif

  llvm::ArrayRef<uint8_t> bytes = python_bytes.GetBytes();
  EXPECT_EQ(bytes.size(), strlen(test_bytes));
  EXPECT_EQ(0, ::memcmp(bytes.data(), test_bytes, bytes.size()));
}

TEST_F(PythonDataObjectsTest, TestPythonByteArray) {
  static const char *test_bytes = "PythonDataObjectsTest::TestPythonByteArray";
  llvm::StringRef orig_bytes(test_bytes);
  PyObject *py_bytes =
      PyByteArray_FromStringAndSize(test_bytes, orig_bytes.size());
  EXPECT_TRUE(PythonByteArray::Check(py_bytes));
  PythonByteArray python_bytes(PyRefType::Owned, py_bytes);
  EXPECT_EQ(PyObjectType::ByteArray, python_bytes.GetObjectType());

  llvm::ArrayRef<uint8_t> after_bytes = python_bytes.GetBytes();
  EXPECT_EQ(after_bytes.size(), orig_bytes.size());
  EXPECT_EQ(0, ::memcmp(orig_bytes.data(), test_bytes, orig_bytes.size()));
}

TEST_F(PythonDataObjectsTest, TestPythonString) {
  // Test that strings behave correctly when wrapped by a PythonString.

  static const char *test_string = "PythonDataObjectsTest::TestPythonString1";
  static const char *test_string2 = "PythonDataObjectsTest::TestPythonString2";

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
  PythonString constructed_string(test_string2);
  EXPECT_EQ(test_string2, constructed_string.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonStringToStr) {
  const char *GetString = "PythonDataObjectsTest::TestPythonStringToStr";

  PythonString str(GetString);
  EXPECT_EQ(GetString, str.GetString());

  PythonString str_str = str.Str();
  EXPECT_EQ(GetString, str_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStr) {}

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStructuredInteger) {
  PythonInteger integer(7);
  auto int_sp = integer.CreateStructuredInteger();
  EXPECT_EQ(7U, int_sp->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonStringToStructuredString) {
  static const char *test_string =
      "PythonDataObjectsTest::TestPythonStringToStructuredString";
  PythonString constructed_string(test_string);
  auto string_sp = constructed_string.CreateStructuredString();
  EXPECT_EQ(test_string, string_sp->GetStringValue());
}

TEST_F(PythonDataObjectsTest, TestPythonListValueEquality) {
  // Test that a list which is built through the native
  // Python API behaves correctly when wrapped by a PythonList.
  static const unsigned list_size = 2;
  static const long long_value0 = 5;
  static const char *const string_value1 = "String Index 1";

  PyObject *py_list = PyList_New(2);
  EXPECT_TRUE(PythonList::Check(py_list));
  PythonList list(PyRefType::Owned, py_list);

  PythonObject list_items[list_size];
  list_items[0].Reset(PythonInteger(long_value0));
  list_items[1].Reset(PythonString(string_value1));

  for (unsigned i = 0; i < list_size; ++i)
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
  EXPECT_EQ(string_value1, chk_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonListManipulation) {
  // Test that manipulation of a PythonList behaves correctly when
  // wrapped by a PythonDictionary.

  static const long long_value0 = 5;
  static const char *const string_value1 = "String Index 1";

  PythonList list(PyInitialValue::Empty);
  PythonInteger integer(long_value0);
  PythonString string(string_value1);

  list.AppendItem(integer);
  list.AppendItem(string);
  EXPECT_EQ(2U, list.GetSize());

  // Verify that the values match
  PythonObject chk_value1 = list.GetItemAtIndex(0);
  PythonObject chk_value2 = list.GetItemAtIndex(1);
  EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
  EXPECT_TRUE(PythonString::Check(chk_value2.get()));

  PythonInteger chk_int(PyRefType::Borrowed, chk_value1.get());
  PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

  EXPECT_EQ(long_value0, chk_int.GetInteger());
  EXPECT_EQ(string_value1, chk_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonListToStructuredList) {
  static const long long_value0 = 5;
  static const char *const string_value1 = "String Index 1";

  PythonList list(PyInitialValue::Empty);
  list.AppendItem(PythonInteger(long_value0));
  list.AppendItem(PythonString(string_value1));

  auto array_sp = list.CreateStructuredArray();
  EXPECT_EQ(StructuredData::Type::eTypeInteger,
            array_sp->GetItemAtIndex(0)->GetType());
  EXPECT_EQ(StructuredData::Type::eTypeString,
            array_sp->GetItemAtIndex(1)->GetType());

  auto int_sp = array_sp->GetItemAtIndex(0)->GetAsInteger();
  auto string_sp = array_sp->GetItemAtIndex(1)->GetAsString();

  EXPECT_EQ(long_value0, long(int_sp->GetValue()));
  EXPECT_EQ(string_value1, string_sp->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonTupleSize) {
  PythonTuple tuple(PyInitialValue::Empty);
  EXPECT_EQ(0U, tuple.GetSize());

  tuple = PythonTuple(3);
  EXPECT_EQ(3U, tuple.GetSize());
}

TEST_F(PythonDataObjectsTest, TestPythonTupleValues) {
  PythonTuple tuple(3);

  PythonInteger int_value(1);
  PythonString string_value("Test");
  PythonObject none_value(PyRefType::Borrowed, Py_None);

  tuple.SetItemAtIndex(0, int_value);
  tuple.SetItemAtIndex(1, string_value);
  tuple.SetItemAtIndex(2, none_value);

  EXPECT_EQ(tuple.GetItemAtIndex(0).get(), int_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(1).get(), string_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(2).get(), none_value.get());
}

TEST_F(PythonDataObjectsTest, TestPythonTupleInitializerList) {
  PythonInteger int_value(1);
  PythonString string_value("Test");
  PythonObject none_value(PyRefType::Borrowed, Py_None);
  PythonTuple tuple{int_value, string_value, none_value};
  EXPECT_EQ(3U, tuple.GetSize());

  EXPECT_EQ(tuple.GetItemAtIndex(0).get(), int_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(1).get(), string_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(2).get(), none_value.get());
}

TEST_F(PythonDataObjectsTest, TestPythonTupleInitializerList2) {
  PythonInteger int_value(1);
  PythonString string_value("Test");
  PythonObject none_value(PyRefType::Borrowed, Py_None);

  PythonTuple tuple{int_value.get(), string_value.get(), none_value.get()};
  EXPECT_EQ(3U, tuple.GetSize());

  EXPECT_EQ(tuple.GetItemAtIndex(0).get(), int_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(1).get(), string_value.get());
  EXPECT_EQ(tuple.GetItemAtIndex(2).get(), none_value.get());
}

TEST_F(PythonDataObjectsTest, TestPythonTupleToStructuredList) {
  PythonInteger int_value(1);
  PythonString string_value("Test");

  PythonTuple tuple{int_value.get(), string_value.get()};

  auto array_sp = tuple.CreateStructuredArray();
  EXPECT_EQ(tuple.GetSize(), array_sp->GetSize());
  EXPECT_EQ(StructuredData::Type::eTypeInteger,
            array_sp->GetItemAtIndex(0)->GetType());
  EXPECT_EQ(StructuredData::Type::eTypeString,
            array_sp->GetItemAtIndex(1)->GetType());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryValueEquality) {
  // Test that a dictionary which is built through the native
  // Python API behaves correctly when wrapped by a PythonDictionary.
  static const unsigned dict_entries = 2;
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

  for (unsigned i = 0; i < dict_entries; ++i)
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
  EXPECT_EQ(value_1, chk_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryManipulation) {
  // Test that manipulation of a dictionary behaves correctly when wrapped
  // by a PythonDictionary.
  static const unsigned dict_entries = 2;

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
  EXPECT_EQ(value_1, chk_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonDictionaryToStructuredDictionary) {
  static const char *const string_key0 = "String Key 0";
  static const char *const string_key1 = "String Key 1";

  static const char *const string_value0 = "String Value 0";
  static const long int_value1 = 7;

  PythonDictionary dict(PyInitialValue::Empty);
  dict.SetItemForKey(PythonString(string_key0), PythonString(string_value0));
  dict.SetItemForKey(PythonString(string_key1), PythonInteger(int_value1));

  auto dict_sp = dict.CreateStructuredDictionary();
  EXPECT_EQ(2U, dict_sp->GetSize());

  EXPECT_TRUE(dict_sp->HasKey(string_key0));
  EXPECT_TRUE(dict_sp->HasKey(string_key1));

  auto string_sp = dict_sp->GetValueForKey(string_key0)->GetAsString();
  auto int_sp = dict_sp->GetValueForKey(string_key1)->GetAsInteger();

  EXPECT_EQ(string_value0, string_sp->GetValue());
  EXPECT_EQ(int_value1, long(int_sp->GetValue()));
}

TEST_F(PythonDataObjectsTest, TestPythonCallableCheck) {
  PythonObject sys_exc_info = m_sys_module.ResolveName("exc_info");
  PythonObject none(PyRefType::Borrowed, Py_None);

  EXPECT_TRUE(PythonCallable::Check(sys_exc_info.get()));
  EXPECT_FALSE(PythonCallable::Check(none.get()));
}

TEST_F(PythonDataObjectsTest, TestPythonCallableInvoke) {
  auto list = m_builtins_module.ResolveName("list").AsType<PythonCallable>();
  PythonInteger one(1);
  PythonString two("two");
  PythonTuple three = {one, two};

  PythonTuple tuple_to_convert = {one, two, three};
  PythonObject result = list({tuple_to_convert});

  EXPECT_TRUE(PythonList::Check(result.get()));
  auto list_result = result.AsType<PythonList>();
  EXPECT_EQ(3U, list_result.GetSize());
  EXPECT_EQ(one.get(), list_result.GetItemAtIndex(0).get());
  EXPECT_EQ(two.get(), list_result.GetItemAtIndex(1).get());
  EXPECT_EQ(three.get(), list_result.GetItemAtIndex(2).get());
}

TEST_F(PythonDataObjectsTest, TestPythonFile) {
  File file(FileSystem::DEV_NULL, File::eOpenOptionRead);
  PythonFile py_file(file, "r");
  EXPECT_TRUE(PythonFile::Check(py_file.get()));
}

TEST_F(PythonDataObjectsTest, TestObjectAttributes) {
  PythonInteger py_int(42);
  EXPECT_TRUE(py_int.HasAttribute("numerator"));
  EXPECT_FALSE(py_int.HasAttribute("this_should_not_exist"));

  PythonInteger numerator_attr =
      py_int.GetAttributeValue("numerator").AsType<PythonInteger>();
  EXPECT_TRUE(numerator_attr.IsAllocated());
  EXPECT_EQ(42, numerator_attr.GetInteger());
}

TEST_F(PythonDataObjectsTest, TestExtractingUInt64ThroughStructuredData) {
  // Make up a custom dictionary with "sys" pointing to the `sys` module.
  const char *key_name = "addr";
  const uint64_t value = 0xf000000000000000ull;
  PythonDictionary python_dict(PyInitialValue::Empty);
  PythonInteger python_ull_value(PyRefType::Owned,
                                 PyLong_FromUnsignedLongLong(value));
  python_dict.SetItemForKey(PythonString(key_name), python_ull_value);
  StructuredData::ObjectSP structured_data_sp =
      python_dict.CreateStructuredObject();
  EXPECT_TRUE((bool)structured_data_sp);
  if (structured_data_sp) {
    StructuredData::Dictionary *structured_dict_ptr =
        structured_data_sp->GetAsDictionary();
    EXPECT_TRUE(structured_dict_ptr != nullptr);
    if (structured_dict_ptr) {
      StructuredData::ObjectSP structured_addr_value_sp =
          structured_dict_ptr->GetValueForKey(key_name);
      EXPECT_TRUE((bool)structured_addr_value_sp);
      const uint64_t extracted_value =
          structured_addr_value_sp->GetIntegerValue(123);
      EXPECT_TRUE(extracted_value == value);
    }
  }
}
