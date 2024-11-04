//===-- PythonDataObjectsTests.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ScriptInterpreter/Python/lldb-python.h"
#include "gtest/gtest.h"

#include "Plugins/ScriptInterpreter/Python/PythonDataObjects.h"
#include "Plugins/ScriptInterpreter/Python/ScriptInterpreterPython.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"

#include "PythonTestSuite.h"

#include <variant>

using namespace lldb_private;
using namespace lldb_private::python;
using llvm::Expected;

class PythonDataObjectsTest : public PythonTestSuite {
public:
  void SetUp() override {
    PythonTestSuite::SetUp();

    m_sys_module = unwrapIgnoringErrors(PythonModule::Import("sys"));
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
  PyObject *obj = PyBytes_FromString("foo");
  Py_ssize_t original_refcnt = Py_REFCNT(obj);
  EXPECT_LE(1, original_refcnt);

  // If we take an owned reference, the refcount should be the same
  PythonObject owned(PyRefType::Owned, obj);
  Py_ssize_t owned_refcnt = Py_REFCNT(owned.get());
  EXPECT_EQ(original_refcnt, owned_refcnt);

  // Take another reference and verify that the refcount increases by 1
  PythonObject strong_ref(owned);
  Py_ssize_t strong_refcnt = Py_REFCNT(strong_ref.get());
  EXPECT_EQ(original_refcnt + 1, strong_refcnt);

  // If we reset the first one, the refcount should be the original value.
  owned.Reset();
  strong_refcnt = Py_REFCNT(strong_ref.get());
  EXPECT_EQ(original_refcnt, strong_refcnt);
}

TEST_F(PythonDataObjectsTest, TestResetting) {
  PythonDictionary dict(PyInitialValue::Empty);

  PyObject *new_dict = PyDict_New();
  dict = Take<PythonDictionary>(new_dict);
  EXPECT_EQ(new_dict, dict.get());

  dict = Take<PythonDictionary>(PyDict_New());
  EXPECT_NE(nullptr, dict.get());
  dict.Reset();
  EXPECT_EQ(nullptr, dict.get());
}

TEST_F(PythonDataObjectsTest, TestBorrowedReferences) {
  PythonByteArray byte_value(PyRefType::Owned,
                             PyByteArray_FromStringAndSize("foo", 3));
  Py_ssize_t original_refcnt = Py_REFCNT(byte_value.get());
  EXPECT_LE(1, original_refcnt);

  PythonByteArray borrowed_byte(PyRefType::Borrowed, byte_value.get());
  Py_ssize_t borrowed_refcnt = Py_REFCNT(borrowed_byte.get());

  EXPECT_EQ(original_refcnt + 1, borrowed_refcnt);
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

  auto major_version_value = As<long long>(major_version_field);
  auto minor_version_value = As<long long>(minor_version_field);

  EXPECT_THAT_EXPECTED(major_version_value, llvm::HasValue(PY_MAJOR_VERSION));
  EXPECT_THAT_EXPECTED(minor_version_value, llvm::HasValue(PY_MINOR_VERSION));
}

TEST_F(PythonDataObjectsTest, TestGlobalNameResolutionWithDot) {
  PythonObject sys_path = m_main_module.ResolveName("sys.path");
  EXPECT_TRUE(sys_path.IsAllocated());
  EXPECT_TRUE(PythonList::Check(sys_path.get()));

  auto version_major =
      As<long long>(m_main_module.ResolveName("sys.version_info.major"));

  auto version_minor =
      As<long long>(m_main_module.ResolveName("sys.version_info.minor"));

  EXPECT_THAT_EXPECTED(version_major, llvm::HasValue(PY_MAJOR_VERSION));
  EXPECT_THAT_EXPECTED(version_minor, llvm::HasValue(PY_MINOR_VERSION));
}

TEST_F(PythonDataObjectsTest, TestDictionaryResolutionWithDot) {
  // Make up a custom dictionary with "sys" pointing to the `sys` module.
  PythonDictionary dict(PyInitialValue::Empty);
  dict.SetItemForKey(PythonString("sys"), m_sys_module);

  // Now use that dictionary to resolve `sys.version_info.major`
  auto version_major = As<long long>(
      PythonObject::ResolveNameWithDictionary("sys.version_info.major", dict));

  auto version_minor = As<long long>(
      PythonObject::ResolveNameWithDictionary("sys.version_info.minor", dict));

  EXPECT_THAT_EXPECTED(version_major, llvm::HasValue(PY_MAJOR_VERSION));
  EXPECT_THAT_EXPECTED(version_minor, llvm::HasValue(PY_MINOR_VERSION));
}

TEST_F(PythonDataObjectsTest, TestPythonInteger) {
  // Test that integers behave correctly when wrapped by a PythonInteger.

  // Verify that `PythonInteger` works correctly when given a PyLong object.
  PyObject *py_long = PyLong_FromLong(12);
  EXPECT_TRUE(PythonInteger::Check(py_long));
  PythonInteger python_long(PyRefType::Owned, py_long);
  EXPECT_EQ(PyObjectType::Integer, python_long.GetObjectType());

  // Verify that you can reset the value and that it is reflected properly.
  python_long.SetInteger(40);
  auto e = As<long long>(python_long);
  EXPECT_THAT_EXPECTED(e, llvm::HasValue(40));

  // Test that creating a `PythonInteger` object works correctly with the
  // int constructor.
  PythonInteger constructed_int(7);
  auto value = As<long long>(constructed_int);
  EXPECT_THAT_EXPECTED(value, llvm::HasValue(7));
}

TEST_F(PythonDataObjectsTest, TestPythonBoolean) {
  // Test PythonBoolean constructed from Py_True
  EXPECT_TRUE(PythonBoolean::Check(Py_True));
  PythonBoolean python_true(PyRefType::Owned, Py_True);
  EXPECT_EQ(PyObjectType::Boolean, python_true.GetObjectType());

  // Test PythonBoolean constructed from Py_False
  EXPECT_TRUE(PythonBoolean::Check(Py_False));
  PythonBoolean python_false(PyRefType::Owned, Py_False);
  EXPECT_EQ(PyObjectType::Boolean, python_false.GetObjectType());

  auto test_from_long = [](long value) {
    PyObject *py_bool = PyBool_FromLong(value);
    EXPECT_TRUE(PythonBoolean::Check(py_bool));
    PythonBoolean python_boolean(PyRefType::Owned, py_bool);
    EXPECT_EQ(PyObjectType::Boolean, python_boolean.GetObjectType());
    EXPECT_EQ(bool(value), python_boolean.GetValue());
  };

  // Test PythonBoolean constructed from long integer values.
  test_from_long(0); // Test 'false' value.
  test_from_long(1); // Test 'true' value.
  test_from_long(~0); // Any value != 0 is 'true'.
}

TEST_F(PythonDataObjectsTest, TestPythonBytes) {
  static const char *test_bytes = "PythonDataObjectsTest::TestPythonBytes";
  PyObject *py_bytes = PyBytes_FromString(test_bytes);
  EXPECT_TRUE(PythonBytes::Check(py_bytes));
  PythonBytes python_bytes(PyRefType::Owned, py_bytes);

  EXPECT_FALSE(PythonString::Check(py_bytes));
  EXPECT_EQ(PyObjectType::Bytes, python_bytes.GetObjectType());

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

  // Verify that `PythonString` works correctly when given a PyUnicode object.
  PyObject *py_unicode = PyUnicode_FromString(test_string);
  EXPECT_TRUE(PythonString::Check(py_unicode));
  PythonString python_unicode(PyRefType::Owned, py_unicode);
  EXPECT_EQ(PyObjectType::String, python_unicode.GetObjectType());
  EXPECT_STREQ(test_string, python_unicode.GetString().data());

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

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStructuredUnsignedInteger) {
  PythonInteger integer(7);
  auto int_sp = integer.CreateStructuredInteger();
  EXPECT_TRUE(
      std::holds_alternative<StructuredData::UnsignedIntegerSP>(int_sp));
  StructuredData::UnsignedIntegerSP uint_sp =
      std::get<StructuredData::UnsignedIntegerSP>(int_sp);
  EXPECT_EQ(7U, uint_sp->GetValue());
}

TEST_F(PythonDataObjectsTest, TestPythonIntegerToStructuredSignedInteger) {
  PythonInteger integer(-42);
  auto int_sp = integer.CreateStructuredInteger();
  EXPECT_TRUE(std::holds_alternative<StructuredData::SignedIntegerSP>(int_sp));
  StructuredData::SignedIntegerSP sint_sp =
      std::get<StructuredData::SignedIntegerSP>(int_sp);
  EXPECT_EQ(-42, sint_sp->GetValue());
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
  list_items[0] = PythonInteger(long_value0);
  list_items[1] = PythonString(string_value1);

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

  auto chkint = As<long long>(chk_value1);
  ASSERT_THAT_EXPECTED(chkint, llvm::HasValue(long_value0));
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

  auto e = As<long long>(chk_int);
  EXPECT_THAT_EXPECTED(e, llvm::HasValue(long_value0));
  EXPECT_EQ(string_value1, chk_str.GetString());
}

TEST_F(PythonDataObjectsTest, TestPythonListToStructuredList) {
  static const long long_value0 = 5;
  static const char *const string_value1 = "String Index 1";

  PythonList list(PyInitialValue::Empty);
  list.AppendItem(PythonInteger(long_value0));
  list.AppendItem(PythonString(string_value1));

  auto array_sp = list.CreateStructuredArray();
  EXPECT_EQ(lldb::eStructuredDataTypeInteger,
            array_sp->GetItemAtIndex(0)->GetType());
  EXPECT_EQ(lldb::eStructuredDataTypeString,
            array_sp->GetItemAtIndex(1)->GetType());

  auto int_sp = array_sp->GetItemAtIndex(0)->GetAsUnsignedInteger();
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
  EXPECT_EQ(lldb::eStructuredDataTypeInteger,
            array_sp->GetItemAtIndex(0)->GetType());
  EXPECT_EQ(lldb::eStructuredDataTypeString,
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

  py_keys[0] = PythonString(key_0);
  py_keys[1] = PythonInteger(key_1);
  py_values[0] = PythonInteger(value_0);
  py_values[1] = PythonString(value_1);

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

  PythonString chk_str(PyRefType::Borrowed, chk_value2.get());
  auto chkint = As<long long>(chk_value1);

  EXPECT_THAT_EXPECTED(chkint, llvm::HasValue(value_0));
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

  keys[0] = PythonString(key_0);
  keys[1] = PythonString(key_1);
  values[0] = PythonInteger(value_0);
  values[1] = PythonString(value_1);

  PythonDictionary dict(PyInitialValue::Empty);
  for (int i = 0; i < 2; ++i)
    dict.SetItemForKey(keys[i], values[i]);

  EXPECT_EQ(dict_entries, dict.GetSize());
  EXPECT_FALSE(dict.HasKey("not_in_dict"));
  EXPECT_TRUE(dict.HasKey(key_0));
  EXPECT_TRUE(dict.HasKey(key_1));

  // Verify that the keys and values match
  PythonObject chk_value1 = dict.GetItemForKey(keys[0]);
  PythonObject chk_value2 = dict.GetItemForKey(keys[1]);
  EXPECT_TRUE(PythonInteger::Check(chk_value1.get()));
  EXPECT_TRUE(PythonString::Check(chk_value2.get()));

  auto chkint = As<long long>(chk_value1);
  PythonString chk_str(PyRefType::Borrowed, chk_value2.get());

  EXPECT_THAT_EXPECTED(chkint, llvm::HasValue(value_0));
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
  auto int_sp = dict_sp->GetValueForKey(string_key1)->GetAsUnsignedInteger();

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
  auto file = FileSystem::Instance().Open(FileSpec(FileSystem::DEV_NULL),
                                          File::eOpenOptionReadOnly);
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  auto py_file = PythonFile::FromFile(*file.get(), "r");
  ASSERT_THAT_EXPECTED(py_file, llvm::Succeeded());
  EXPECT_TRUE(PythonFile::Check(py_file.get().get()));
}

TEST_F(PythonDataObjectsTest, TestObjectAttributes) {
  PythonInteger py_int(42);
  EXPECT_TRUE(py_int.HasAttribute("numerator"));
  EXPECT_FALSE(py_int.HasAttribute("this_should_not_exist"));

  auto numerator_attr = As<long long>(py_int.GetAttributeValue("numerator"));

  EXPECT_THAT_EXPECTED(numerator_attr, llvm::HasValue(42));
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
          structured_addr_value_sp->GetUnsignedIntegerValue(123);
      EXPECT_TRUE(extracted_value == value);
    }
  }
}

TEST_F(PythonDataObjectsTest, TestCallable) {

  PythonDictionary globals(PyInitialValue::Empty);
  auto builtins = PythonModule::BuiltinsModule();
  llvm::Error error = globals.SetItem("__builtins__", builtins);
  ASSERT_FALSE(error);

  {
    PyObject *o = PyRun_String("lambda x : x", Py_eval_input, globals.get(),
                               globals.get());
    ASSERT_FALSE(o == NULL);
    auto lambda = Take<PythonCallable>(o);
    auto arginfo = lambda.GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);
  }

  {
    PyObject *o = PyRun_String("lambda x,y=0: x", Py_eval_input, globals.get(),
                               globals.get());
    ASSERT_FALSE(o == NULL);
    auto lambda = Take<PythonCallable>(o);
    auto arginfo = lambda.GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 2u);
  }

  {
    PyObject *o = PyRun_String("lambda x,y=0, **kw: x", Py_eval_input,
                               globals.get(), globals.get());
    ASSERT_FALSE(o == NULL);
    auto lambda = Take<PythonCallable>(o);
    auto arginfo = lambda.GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 2u);
  }

  {
    PyObject *o = PyRun_String("lambda x,y,*a: x", Py_eval_input, globals.get(),
                               globals.get());
    ASSERT_FALSE(o == NULL);
    auto lambda = Take<PythonCallable>(o);
    auto arginfo = lambda.GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args,
              PythonCallable::ArgInfo::UNBOUNDED);
  }

  {
    PyObject *o = PyRun_String("lambda x,y,*a,**kw: x", Py_eval_input,
                               globals.get(), globals.get());
    ASSERT_FALSE(o == NULL);
    auto lambda = Take<PythonCallable>(o);
    auto arginfo = lambda.GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args,
              PythonCallable::ArgInfo::UNBOUNDED);
  }

  {
    const char *script = R"(
class Foo:
  def bar(self, x):
     return x
  @classmethod
  def classbar(cls, x):
     return x
  @staticmethod
  def staticbar(x):
     return x
  def __call__(self, x):
     return x
obj = Foo()
bar_bound   = Foo().bar
bar_class   = Foo().classbar
bar_static  = Foo().staticbar
bar_unbound = Foo.bar


class OldStyle:
  def __init__(self, one, two, three):
    pass

class NewStyle(object):
  def __init__(self, one, two, three):
    pass

)";
    PyObject *o =
        PyRun_String(script, Py_file_input, globals.get(), globals.get());
    ASSERT_FALSE(o == NULL);
    Take<PythonObject>(o);

    auto bar_bound = As<PythonCallable>(globals.GetItem("bar_bound"));
    ASSERT_THAT_EXPECTED(bar_bound, llvm::Succeeded());
    auto arginfo = bar_bound.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);

    auto bar_unbound = As<PythonCallable>(globals.GetItem("bar_unbound"));
    ASSERT_THAT_EXPECTED(bar_unbound, llvm::Succeeded());
    arginfo = bar_unbound.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 2u);

    auto bar_class = As<PythonCallable>(globals.GetItem("bar_class"));
    ASSERT_THAT_EXPECTED(bar_class, llvm::Succeeded());
    arginfo = bar_class.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);

    auto bar_static = As<PythonCallable>(globals.GetItem("bar_static"));
    ASSERT_THAT_EXPECTED(bar_static, llvm::Succeeded());
    arginfo = bar_static.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);

    auto obj = As<PythonCallable>(globals.GetItem("obj"));
    ASSERT_THAT_EXPECTED(obj, llvm::Succeeded());
    arginfo = obj.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);

    auto oldstyle = As<PythonCallable>(globals.GetItem("OldStyle"));
    ASSERT_THAT_EXPECTED(oldstyle, llvm::Succeeded());
    arginfo = oldstyle.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 3u);

    auto newstyle = As<PythonCallable>(globals.GetItem("NewStyle"));
    ASSERT_THAT_EXPECTED(newstyle, llvm::Succeeded());
    arginfo = newstyle.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 3u);
  }

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 3

  // the old implementation of GetArgInfo just doesn't work on builtins.

  {
    auto builtins = PythonModule::BuiltinsModule();
    auto hex = As<PythonCallable>(builtins.GetAttribute("hex"));
    ASSERT_THAT_EXPECTED(hex, llvm::Succeeded());
    auto arginfo = hex.get().GetArgInfo();
    ASSERT_THAT_EXPECTED(arginfo, llvm::Succeeded());
    EXPECT_EQ(arginfo.get().max_positional_args, 1u);
  }

#endif
}

TEST_F(PythonDataObjectsTest, TestScript) {

  static const char script[] = R"(
def factorial(n):
  if n > 1:
    return n * factorial(n-1)
  else:
    return 1;
main = factorial
)";

  PythonScript factorial(script);

  EXPECT_THAT_EXPECTED(As<long long>(factorial(5ll)), llvm::HasValue(120));
}

TEST_F(PythonDataObjectsTest, TestExceptions) {

  static const char script[] = R"(
def foo():
  return bar()
def bar():
  return baz()
def baz():
  return 1 / 0
main = foo
)";

  PythonScript foo(script);

  EXPECT_THAT_EXPECTED(
      foo(), llvm::Failed<PythonException>(testing::Property(
                 &PythonException::ReadBacktrace,
                 testing::AllOf(testing::ContainsRegex("line 3, in foo"),
                                testing::ContainsRegex("line 5, in bar"),
                                testing::ContainsRegex("line 7, in baz"),
                                testing::ContainsRegex("ZeroDivisionError")))));

#if !((defined(_WIN32) || defined(_WIN64)) && (defined(__aarch64__) || defined(_M_ARM64)))

  static const char script2[] = R"(
class MyError(Exception):
  def __str__(self):
    return self.my_message

def main():
  raise MyError("lol")

)";

  PythonScript lol(script2);

  EXPECT_THAT_EXPECTED(
      lol(),
      llvm::Failed<PythonException>(testing::Property(
          &PythonException::ReadBacktrace,
          testing::AnyOf(
              testing::ContainsRegex("MyError: <exception str\\(\\) failed>"),
              testing::ContainsRegex("unprintable MyError")))));

#endif
}

TEST_F(PythonDataObjectsTest, TestRun) {

  PythonDictionary globals(PyInitialValue::Empty);

  auto x = As<long long>(runStringOneLine("40 + 2", globals, globals));
  ASSERT_THAT_EXPECTED(x, llvm::Succeeded());
  EXPECT_EQ(x.get(), 42l);

  Expected<PythonObject> r = runStringOneLine("n = 42", globals, globals);
  ASSERT_THAT_EXPECTED(r, llvm::Succeeded());
  auto y = As<long long>(globals.GetItem("n"));
  ASSERT_THAT_EXPECTED(y, llvm::Succeeded());
  EXPECT_EQ(y.get(), 42l);

  const char script[] = R"(
def foobar():
  return "foo" + "bar" + "baz"
g = foobar()
)";

  r = runStringMultiLine(script, globals, globals);
  ASSERT_THAT_EXPECTED(r, llvm::Succeeded());
  auto g = As<std::string>(globals.GetItem("g"));
  ASSERT_THAT_EXPECTED(g, llvm::HasValue("foobarbaz"));
}
