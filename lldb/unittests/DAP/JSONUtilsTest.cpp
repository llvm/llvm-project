//===-- JSONUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONUtils.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;

TEST(JSONUtilsTest, GetAsString) {
  json::Value string_value("foo");
  EXPECT_EQ(GetAsString(string_value), "foo");

  json::Value int_value(42);
  EXPECT_EQ(GetAsString(int_value), "");

  json::Value null_value(nullptr);
  EXPECT_EQ(GetAsString(null_value), "");
}

TEST(JSONUtilsTest, GetString_Ref) {
  json::Object obj;
  obj.try_emplace("key", "value");

  auto result = GetString(obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "value");

  result = GetString(obj, "nonexistent_key");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetString_Pointer) {
  json::Object obj;
  obj.try_emplace("key", "value");

  auto result = GetString(&obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "value");

  result = GetString(nullptr, "key");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetBoolean_Ref) {
  json::Object obj;
  obj.try_emplace("key_true", true);
  obj.try_emplace("key_false", false);
  obj.try_emplace("key_int", 1);

  auto result = GetBoolean(obj, "key_true");
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());

  result = GetBoolean(obj, "key_false");
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result.value());

  result = GetBoolean(obj, "key_int");
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());

  result = GetBoolean(obj, "nonexistent_key");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetBoolean_Pointer) {
  json::Object obj;
  obj.try_emplace("key", true);

  auto result = GetBoolean(&obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());

  result = GetBoolean(nullptr, "key");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetInteger_Ref) {
  json::Object obj;
  obj.try_emplace("key", 123);

  auto result = GetInteger<int>(obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 123);

  result = GetInteger<int>(obj, "nonexistent_key");
  EXPECT_FALSE(result.has_value());

  obj.try_emplace("key_float", 123.45);
  result = GetInteger<int>(obj, "key_float");
  EXPECT_FALSE(result.has_value());

  obj.try_emplace("key_string", "123");
  result = GetInteger<int>(obj, "key_string");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetInteger_Pointer) {
  json::Object obj;
  obj.try_emplace("key", 456);

  auto result = GetInteger<int>(&obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 456);

  result = GetInteger<int>(nullptr, "key");
  EXPECT_FALSE(result.has_value());

  obj.try_emplace("key_invalid", "not_an_integer");
  result = GetInteger<int>(&obj, "key_invalid");
  EXPECT_FALSE(result.has_value());
}

TEST(JSONUtilsTest, GetInteger_DifferentTypes) {
  json::Object obj;
  obj.try_emplace("key", 789);

  auto result = GetInteger<int64_t>(obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 789);

  result = GetInteger<uint32_t>(obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 789U);

  result = GetInteger<int16_t>(obj, "key");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), static_cast<int16_t>(789));
}

TEST(JSONUtilsTest, GetStrings_EmptyArray) {
  llvm::json::Object obj;
  obj.try_emplace("key", llvm::json::Array());
  auto result = GetStrings(&obj, "key");
  EXPECT_TRUE(result.empty());
}

TEST(JSONUtilsTest, GetStrings_NullKey) {
  llvm::json::Object obj;
  auto result = GetStrings(&obj, "nonexistent_key");
  EXPECT_TRUE(result.empty());
}

TEST(JSONUtilsTest, GetStrings_StringValues) {
  llvm::json::Object obj;
  llvm::json::Array arr{"value1", "value2", "value3"};
  obj.try_emplace("key", std::move(arr));
  auto result = GetStrings(&obj, "key");
  ASSERT_EQ(result.size(), 3UL);
  EXPECT_EQ(result[0], "value1");
  EXPECT_EQ(result[1], "value2");
  EXPECT_EQ(result[2], "value3");
}

TEST(JSONUtilsTest, GetStrings_MixedValues) {
  llvm::json::Object obj;
  llvm::json::Array arr{"string", 42, true, nullptr};
  obj.try_emplace("key", std::move(arr));
  auto result = GetStrings(&obj, "key");
  ASSERT_EQ(result.size(), 3UL);
  EXPECT_EQ(result[0], "string");
  EXPECT_EQ(result[1], "42");
  EXPECT_EQ(result[2], "true");
}

TEST(JSONUtilsTest, GetStrings_NestedArray) {
  llvm::json::Object obj;
  llvm::json::Array nested_array{"string", llvm::json::Array{"nested"}};
  obj.try_emplace("key", std::move(nested_array));
  auto result = GetStrings(&obj, "key");
  ASSERT_EQ(result.size(), 1UL);
  EXPECT_EQ(result[0], "string");
}

TEST(JSONUtilsTest, DecodeMemoryReference) {
  EXPECT_EQ(DecodeMemoryReference(""), std::nullopt);
  EXPECT_EQ(DecodeMemoryReference("123"), std::nullopt);
  EXPECT_EQ(DecodeMemoryReference("0o123"), std::nullopt);
  EXPECT_EQ(DecodeMemoryReference("0b1010101"), std::nullopt);
  EXPECT_EQ(DecodeMemoryReference("0x123"), 291u);

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_TRUE(DecodeMemoryReference(json::Object{{"mem_ref", "0x123"}},
                                      "mem_ref", addr, root,
                                      /*required=*/true));
    EXPECT_EQ(addr, 291u);
  }

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_TRUE(DecodeMemoryReference(json::Object{}, "mem_ref", addr, root,
                                      /*required=*/false));
  }

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_FALSE(DecodeMemoryReference(json::Value{"string"}, "mem_ref", addr,
                                       root,
                                       /*required=*/true));
    EXPECT_THAT_ERROR(root.getError(), FailedWithMessage("expected object"));
  }

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_FALSE(DecodeMemoryReference(json::Object{}, "mem_ref", addr, root,
                                       /*required=*/true));
    EXPECT_THAT_ERROR(root.getError(),
                      FailedWithMessage("missing value at (root).mem_ref"));
  }

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_FALSE(DecodeMemoryReference(json::Object{{"mem_ref", 123}},
                                       "mem_ref", addr, root,
                                       /*required=*/true));
    EXPECT_THAT_ERROR(root.getError(),
                      FailedWithMessage("expected string at (root).mem_ref"));
  }

  {
    addr_t addr = LLDB_INVALID_ADDRESS;
    json::Path::Root root;
    EXPECT_FALSE(DecodeMemoryReference(json::Object{{"mem_ref", "123"}},
                                       "mem_ref", addr, root,
                                       /*required=*/true));
    EXPECT_THAT_ERROR(
        root.getError(),
        FailedWithMessage("malformed memory reference at (root).mem_ref"));
  }
}
