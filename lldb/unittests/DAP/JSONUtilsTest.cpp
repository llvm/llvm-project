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
