//===-- JSONUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONUtils.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBTarget.h"
#include "llvm/Support/JSON.h"
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

TEST(JSONUtilsTest, CreateModule) {
  SBTarget target;
  SBModule module;

  json::Value value = CreateModule(target, module);
  json::Object *object = value.getAsObject();

  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->size(), 0UL);
}
