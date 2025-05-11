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
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;

TEST(JSONUtilsTest, GetAsString) {
  StringRef str = "foo";
  json::Value value("foo");
  EXPECT_EQ(str, GetAsString(value));
}

TEST(JSONUtilsTest, CreateModule) {
  SBTarget target;
  SBModule module;

  json::Value value = CreateModule(target, module);
  json::Object *object = value.getAsObject();

  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->size(), 0UL);
}
