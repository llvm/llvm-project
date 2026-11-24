//===-- StructuredDataTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestingSupport/TestUtilities.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/Support/Path.h"

using namespace lldb;
using namespace lldb_private;

TEST(StructuredDataTest, StringDump) {
  std::pair<llvm::StringRef, llvm::StringRef> TestCases[] = {
      {R"(asdfg)", R"("asdfg")"},
      {R"(as"df)", R"("as\"df")"},
      {R"(as\df)", R"("as\\df")"},
  };
  for (auto P : TestCases) {
    StreamString S;
    const bool pretty_print = false;
    StructuredData::String(P.first).Dump(S, pretty_print);
    EXPECT_EQ(P.second, S.GetString());
  }
}

TEST(StructuredDataTest, GetDescriptionEmpty) {
  Status status;
  auto object_sp = StructuredData::ParseJSON("{}");
  ASSERT_NE(nullptr, object_sp);

  StreamString S;
  object_sp->GetDescription(S);
  EXPECT_EQ(0u, S.GetSize());
}

TEST(StructuredDataTest, GetDescriptionBasic) {
  Status status;
  std::string input = GetInputFilePath("StructuredData-basic.json");
  auto object_sp = StructuredData::ParseJSONFromFile(FileSpec(input), status);
  ASSERT_NE(nullptr, object_sp);

  const std::string expected = "[0]: 1\n"
                               "[1]: 2\n"
                               "[2]: 3";

  StreamString S;
  object_sp->GetDescription(S);
  EXPECT_EQ(expected, S.GetString());
}

TEST(StructuredDataTest, GetDescriptionNested) {
  Status status;
  std::string input = GetInputFilePath("StructuredData-nested.json");
  auto object_sp = StructuredData::ParseJSONFromFile(FileSpec(input), status);
  ASSERT_NE(nullptr, object_sp);

  const std::string expected = "my_dict:\n"
                               "  [0]:\n"
                               "    three: 3\n"
                               "    two: 2\n"
                               "  [1]:\n"
                               "    four:\n"
                               "      val: 4\n"
                               "  [2]: 1";

  StreamString S;
  object_sp->GetDescription(S);
  EXPECT_EQ(expected, S.GetString());
}

TEST(StructuredDataTest, GetDescriptionFull) {
  Status status;
  std::string input = GetInputFilePath("StructuredData-full.json");
  auto object_sp = StructuredData::ParseJSONFromFile(FileSpec(input), status);
  ASSERT_NE(nullptr, object_sp);

  const std::string expected = "Array:\n"
                               "  [0]: 3.140000\n"
                               "  [1]:\n"
                               "    key: val\n"
                               "Dictionary:\n"
                               "  FalseBool: False\n"
                               "Integer: 1\n"
                               "Null: NULL\n"
                               "String: value\n"
                               "TrueBool: True";

  StreamString S;
  object_sp->GetDescription(S);
  EXPECT_EQ(expected, S.GetString());
}

TEST(StructuredDataTest, ParseJSONFromFile) {
  Status status;
  auto object_sp = StructuredData::ParseJSONFromFile(
      FileSpec("non-existing-file.json"), status);
  EXPECT_EQ(nullptr, object_sp);

  std::string input = GetInputFilePath("StructuredData-basic.json");
  object_sp = StructuredData::ParseJSONFromFile(FileSpec(input), status);
  ASSERT_NE(nullptr, object_sp);

  StreamString S;
  object_sp->Dump(S, false);
  EXPECT_EQ("[1,2,3]", S.GetString());
}

TEST(StructuredDataTest, ParseJSONValue) {
  llvm::json::Value value = llvm::json::Object{
      {"array", llvm::json::Array{1, "two", nullptr}},
      {"bool", true},
      {"dict", llvm::json::Object{{"key", "val"}}},
      {"float", 3.5},
      {"negative", -1},
      {"null", nullptr},
      {"string", "value"},
      {"uint64_max", std::numeric_limits<uint64_t>::max()},
  };
  auto object_sp = StructuredData::ParseJSONValue(value);
  ASSERT_NE(nullptr, object_sp);

  StreamString S;
  object_sp->Dump(S, /*pretty_print=*/false);
  EXPECT_EQ(R"({"array":[1,"two",null],"bool":true,"dict":{"key":"val"},)"
            R"("float":3.5,"negative":-1,"null":null,"string":"value",)"
            R"("uint64_max":18446744073709551615})",
            S.GetString());

  StructuredData::Dictionary *dict = object_sp->GetAsDictionary();
  ASSERT_NE(nullptr, dict);
  EXPECT_EQ(eStructuredDataTypeArray, dict->GetValueForKey("array")->GetType());
  EXPECT_EQ(eStructuredDataTypeBoolean,
            dict->GetValueForKey("bool")->GetType());
  EXPECT_EQ(eStructuredDataTypeDictionary,
            dict->GetValueForKey("dict")->GetType());
  EXPECT_EQ(eStructuredDataTypeFloat, dict->GetValueForKey("float")->GetType());
  EXPECT_EQ(eStructuredDataTypeSignedInteger,
            dict->GetValueForKey("negative")->GetType());
  EXPECT_EQ(eStructuredDataTypeNull, dict->GetValueForKey("null")->GetType());
  EXPECT_EQ(eStructuredDataTypeString,
            dict->GetValueForKey("string")->GetType());
  EXPECT_EQ(eStructuredDataTypeUnsignedInteger,
            dict->GetValueForKey("uint64_max")->GetType());
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(),
            dict->GetValueForKey("uint64_max")->GetUnsignedIntegerValue());
}
